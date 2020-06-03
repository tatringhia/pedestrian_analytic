# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self.matching = []
        self.default_id = 0  # change _next_id to _default_id to assign _default_id for _initiatet_track
        self.assign_id = 1  # unique id assigns to confirmed track

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            if not track.is_deleted():
                track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # Deleted trakcs confirmed is_deleted and not assigned ID
        # Deleted tracks kept, may burden the processor
        self.tracks = [t for t in self.tracks if not (t.is_deleted() and t.track_id == self.default_id)]

        # assign ID to just confirmed tracks and track_id = default_id
        self._assign_id()

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

        self.matching = matches

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        deleted_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_deleted()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if i not in confirmed_tracks and i not in deleted_tracks]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self.default_id, self.n_init, self.max_age,
            detection.feature))

    def _assign_id(self):
        """
        Assign id to tracks and remove deleted tracks re-assigned id
        
        """
        # Get index of just confirmed track
        just_confirmed_indices = [i for i, t in enumerate(self.tracks) if t.is_confirmed() and t.track_id == 0]
        if len(just_confirmed_indices) > 0:
            # get index of deleted tracks
            deleted_indices = [i for i, t in enumerate(self.tracks)
                               if t.is_deleted() and t.track_id in self.metric.samples.keys()]
            # match id of just confirmed index and deleted tracks
            row_col = self._reid(deleted_indices, just_confirmed_indices)
            # Re-assign for just_detection tracks matched to deleted tracks
            if row_col is not None:
                for i, j in row_col:
                    self.tracks[just_confirmed_indices[j]].track_id = self.tracks[deleted_indices[i]].track_id

            # Assign new id for un_matched deleted tracks from external
            for i in just_confirmed_indices:
                if self.tracks[i].track_id == self.default_id:
                    # Calling to external re-id to assign id to confirmed tracks
                    pass

            # if not get id from external, assign new id
            for i in just_confirmed_indices:
                if self.tracks[i].track_id == self.default_id:
                    self.tracks[i].track_id = self.assign_id
                    self.assign_id += 1

            # remove deleted tracks re-assigned id         
            if row_col is not None:
                for i, j in row_col:
                    self.tracks.pop(deleted_indices[i])

    def _reid(self, deleted_indices, just_confirmed_indices):
        """
        Match just confirm tracks to deleted tracks
        
        Parameters:
        -----------
            deleted_indices: list[int], 
                A list of track index is_deleted()
            just_confirmed_indices: list[int] 
                A list of track index is_confirmed and waiting assing id
        
        Return:
        -------
            row_col: list[(row, col)]
                A list of tuples: row ~ index of deleted_indices, col ~ index of just_confirmed_indices
                
        """
        if len(deleted_indices) != 0 and len(just_confirmed_indices) != 0:
            # Create feature matrix M x L, M ~ just confirmed tracks, L ~ 128 (from featrue_extractor)
            features = np.array([[self.tracks[i].features[-1]] for i in just_confirmed_indices])[0]
            targets = np.array([self.tracks[i].track_id for i in deleted_indices])
            # Create empty cost matrix
            # cost_matrix = np.zeros((len(targets), len(features)))
            # Calculate cost (cosine distance) matrix of deleted tracks and just confirmed track
            cost_matrix = self.metric.distance(features, targets)
            # suppress distance > matching threshold
            cost_matrix[cost_matrix > self.metric.matching_threshold] = 1e+5
            # get largest disctance
            row_max = np.argmax(-cost_matrix, axis=0)
            row_col = [(row, i) for i, row in enumerate(row_max)]
            return row_col
        else:
            return None
