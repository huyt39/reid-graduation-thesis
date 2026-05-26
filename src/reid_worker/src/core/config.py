from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "reid_worker"
    # Kafka replay / short-video mode should drain as fast as possible once
    # messages are available. Sleeping between consumed batches only delays the
    # point at which late-frame identities are assigned and persisted.
    poll_interval_s: float = 0.0
    log_every_n_messages: int = 25

    kafka_bootstrap_servers: str = "localhost:29092"
    input_topic: str = "reid_input"
    output_topic: str = "reid_output"
    consumer_group: str = "reid_worker_group"
    schema_path: str = "src/contracts/reid_input.avsc"
    output_schema_path: str = "src/contracts/reid_output.avsc"

    model_service_url: str = "http://localhost:8000"
    # Which embedding model the inference engine should use. "osnet" is the
    # OSNet-x1.0 trained on Market-1501 (single global feature, 512-d).
    # "lmbn" uses Light Multi-Branch Network — multi-part body features
    # (global + horizontal stripes + channel branches), more robust to
    # partial-body / occluded crops. Requires INFERENCE_LMBN_WEIGHTS to be
    # set in the inference engine.
    embedding_model: str = "osnet"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    embedding_dim: int = 512
    similarity_threshold: float = 0.73
    match_margin: float = 0.06
    spatial_reuse_threshold: float = 0.62
    momentum: float = 0.8
    max_gallery_size: int = 8
    recent_person_reuse_enabled: bool = True
    recent_person_reuse_seconds: float = 2.5
    recent_person_reuse_min_iou: float = 0.2
    recent_person_reuse_max_center_distance_ratio: float = 0.75

    tracklet_min_entries: int = 6
    tracklet_max_entries: int = 60
    tracklet_window_seconds: float = 3.0
    tracklet_stale_seconds: float = 5.0
    tracklet_idle_flush_enabled: bool = True
    # Frame-index based continuity memory. This survives short wall-clock
    # stalls/backpressure, where active-track cleanup may remove the live
    # mapping even though the next tracklet is adjacent in video time.
    track_identity_memory_max_gap_frames: int = 180
    track_identity_memory_max_center_distance_ratio: float = 1.25
    # Realtime bias: once a track stops receiving frames, flush quickly so the
    # person/timeline views reflect confirmed identities within sub-second idle
    # gaps instead of waiting multiple seconds for end-of-stream drain.
    tracklet_idle_flush_seconds: float = 0.5

    topk_k: int = 5
    topk_min_temporal_gap: int = 3
    overlap_lambda: float = 0.3
    min_high_quality_frames: int = 3
    high_quality_threshold: float = 0.55
    # PDF Bước 2 — tracklet consistency features (bbox_size_stability,
    # position_stability, good_frame_ratio) act as a coarse
    # mixed-identity filter, not a strict readiness gate. The threshold
    # is applied to consistency.overall (see compute_tracklet_consistency).
    # A person walking diagonally toward the camera produces legitimate
    # bbox growth + centroid drift, with overall sitting around 0.50-0.60;
    # gating above that band rejects real people. 0.35 still catches
    # actual mixed-identity tracklets — those collapse position_stability
    # to ≤ 0.2 during the ID-swap, pulling overall below 0.35 — while
    # letting normal motion through. Tune via the env override per scene.
    tracklet_readiness_consistency_threshold: float = 0.35

    gamma: float = 0.5
    embedding_consensus_threshold: float = 0.66
    min_consensus_embeddings: int = 2
    # Outlier-aware aggregation: after the first weighted mean is computed,
    # drop frames whose cosine sim to that mean is below this threshold and
    # recompute on the survivors. Defends against a single mis-tracked crop
    # skewing the tracklet representation. Set to 0.0 to disable.
    agg_outlier_threshold: float = 0.5

    # Tracklet-build-time appearance gate: when an existing tracklet receives a
    # new high-quality frame (v_worker ≥ this threshold), compare its embedding
    # against the running buffer mean. If cosine sim < tracklet_split_threshold
    # the frame is split off into a virtual track_id rather than poisoning the
    # buffer. Cuts contamination from ByteTrack ID-swaps during occlusion.
    tracklet_appearance_gate_enabled: bool = True
    tracklet_appearance_gate_min_v: float = 0.6
    tracklet_split_threshold: float = 0.55
    # The appearance gate calls the embedding service from the per-frame input
    # path. Checking every high-quality frame makes the worker fall behind video
    # playback; sample periodically after the gate is armed.
    tracklet_appearance_gate_check_interval_frames: int = 6
    # Tracklet-level ID-swap guard. ByteTrack can keep the same track_id while
    # the crop drifts from one person to another during occlusion. Such a
    # tracklet must not be admitted through low-threshold current-identity
    # continuity; keep it as occlusion evidence and wait for cleaner fragments.
    tracklet_identity_shift_guard_enabled: bool = True
    tracklet_identity_shift_min_entries: int = 12
    tracklet_identity_shift_min_endpoint_displacement_ratio: float = 0.55
    tracklet_identity_shift_min_size_ratio: float = 1.60
    tracklet_identity_shift_min_area_ratio: float = 2.20
    tracklet_identity_shift_anchor_min_frame_gap: int = 24
    tracklet_identity_shift_anchor_min_endpoint_displacement_ratio: float = 0.40
    tracklet_identity_shift_anchor_min_size_ratio: float = 1.45
    tracklet_identity_shift_anchor_min_area_ratio: float = 1.90

    # Established-identity merge: similarity threshold required when BOTH
    # candidate persons exceed duplicate_merge_weak_max_tracklets. Held above
    # typical cross-person cosine to prevent collapsing distinct identities,
    # below typical same-person-cross-pose so legitimate duplicates still merge.
    # Existing cooccurrence + attribute guards apply on top.
    duplicate_merge_established_min_score: float = 0.78

    # Background re-merge reconciler: every N seconds, scan recently-updated
    # persons for cross-person gallery similarity and merge duplicates that the
    # weak-only inline merge cannot catch. Set interval to 0 to disable.
    background_reconciler_interval_s: float = 30.0
    background_reconciler_max_persons: int = 50
    final_reconciler_passes: int = 3

    # Stream quiescence: cut off new person_id minting after the worker has
    # been receiving no Kafka traffic for this many seconds. Reference is the
    # worker's wall-clock receipt time (NOT the message's claimed timestamp).
    # 20s is empirical headroom for end-of-stream pipeline drain: tracklets
    # become "ready" 3s after their last frame, then process serially through
    # the async task queue (~1s each, can stack to ~10s for many concurrent
    # tracklets), then a few seconds of safety. Bump higher if your workload
    # has more end-of-stream tracklets than this allows for. Set to 0 to
    # disable the gate entirely (re-introduces the post-stream phantom-person
    # creation bug).
    stream_quiescence_seconds: float = 20.0
    stream_finalization_timeout_seconds: float = 60.0
    # Separate backlog guard for new person_id allocation. This uses the edge
    # publish timestamp on tracklet entries, so if the worker is still chewing
    # through minutes-old Kafka frames after a video ended, it may match existing
    # identities but must not mint new people.
    max_new_identity_lag_seconds: float = 30.0

    # Gender voting hysteresis (PDF Bước 6: vote per tracklet, change label only
    # when two consecutive tracklets agree with high confidence). A person is
    # considered "committed" to a gender when the most recent N high-confidence
    # tracklet sightings (sorted by start time) all agree. A single bad
    # tracklet can no longer trigger persons_have_clear_gender_disagreement.
    gender_tracklet_flip_confidence: float = 0.80
    gender_tracklet_min_consecutive: int = 2
    gender_ambiguous_conflict_enabled: bool = True
    gender_ambiguous_conflict_tracklet_confidence: float = 0.70
    gender_ambiguous_conflict_max_person_confidence: float = 0.80
    # fragment_recovery defaults stay False so the test fixtures and existing
    # deployments aren't surprised; production opts in via WORKER_FRAGMENT_RECOVERY_ENABLED.

    occlusion_candidates_enabled: bool = True
    occlusion_candidate_min_entries: int = 2
    occlusion_candidate_min_visibility: float = 0.45
    untracked_detection_candidates_enabled: bool = True
    untracked_detection_raw_candidates_enabled: bool = False
    untracked_detection_min_confidence: float = 0.25
    untracked_detection_min_visibility: float = 0.35
    untracked_detection_max_track_iou: float = 0.20
    untracked_detection_cluster_enabled: bool = True
    untracked_detection_cluster_min_entries: int = 2
    untracked_detection_cluster_max_gap_frames: int = 15
    untracked_detection_cluster_max_center_distance_ratio: float = 0.95
    # Sliding-window cap on per-cluster TrackletEntry retention. Each entry
    # holds a full-resolution image crop, so unbounded growth OOM-kills the
    # worker. After promotion the tracklet_buffer copy is authoritative; the
    # cluster only retains a recent window so subsequent entries can still
    # extend it if the person stays in view.
    untracked_detection_cluster_max_entries: int = 30
    untracked_detection_cluster_flush_after_frames: int = 36
    # When ByteTrack fails to track a person (small/distant or boundary-crossing),
    # YOLO detections accumulate into untracked_detection_clusters. With promotion
    # enabled, clusters with sufficient evidence are pushed into tracklet_buffer
    # so the normal embedding+matcher pipeline can ReID them. All standard gates
    # (consensus, near_gallery_defer, promote_consistency) still apply.
    untracked_cluster_promote_enabled: bool = True
    untracked_cluster_promote_min_entries: int = 6
    untracked_cluster_promote_min_visibility: float = 0.65
    # High-confidence tier recovers short, clean untracked clusters that
    # ByteTrack missed. It still requires 5 frames plus a strong consistency
    # gate, so 2-4 frame static/object bursts remain occlusion candidates.
    untracked_cluster_promote_min_entries_fast: int = 5
    untracked_cluster_promote_min_visibility_fast: float = 0.85
    untracked_cluster_promote_fast_min_overall_consistency: float = 0.88
    recover_stale_tracklets_enabled: bool = True
    # Keep short occlusion fragments as evidence only by default. Promoting a
    # 2-4 frame fragment to a confirmed identity caused duplicate IDs on
    # occluded/rear-view re-entries, so confirmation must stay behind the
    # normal tracklet matcher or an explicit offline review step.
    fragment_recovery_enabled: bool = False
    fragment_recovery_min_fragments: int = 2
    fragment_recovery_min_total_entries: int = 5
    fragment_recovery_min_visibility: float = 0.72
    fragment_recovery_min_similarity: float = 0.62
    fragment_recovery_max_gap_frames: int = 180
    fragment_recovery_max_center_distance_ratio: float = 1.8
    # Search floor for diagnostics. Any fragment cluster near an existing
    # gallery is deferred instead of minted here; the main matcher should own
    # ambiguous occlusion decisions.
    fragment_recovery_near_gallery_threshold: float = 0.52

    promote_v_threshold: float = 0.6
    promote_consistency_threshold: float = 0.65
    synthetic_new_identity_min_overall_consistency: float = 0.75
    # PDF Bước 2 — "good frame streak" is a tracklet-quality signal. It is
    # recorded and passed through the matcher, but new identity creation still
    # follows the conjunctive delayed-promotion gates from Bước 5.
    good_streak_promotion_enabled: bool = True
    good_streak_min_consecutive: int = 4
    # New identities need enough temporal support to avoid minting duplicates
    # from brief occlusion fragments. Shorter fragments may still attach as
    # provisional occlusion evidence or remain tentative.
    new_identity_min_tracklet_len: int = 6

    tentative_max_attempts: int = 5
    # Kept True because borderline-quality persons (partial body, occluded)
    # often never clear promote_v_threshold + promote_consistency_threshold
    # cleanly and rely on the fallback path for their first ID. The
    # stream_quiescence_seconds gate above is what prevents this path from
    # producing phantom persons long after the stream ends; together they
    # give legitimate borderline persons an ID while still cutting the leak.
    tentative_fallback_enabled: bool = True

    update_v_threshold: float = 0.6
    update_consistency_threshold: float = 0.7
    update_min_tracklet_len: int = 5
    update_sim_threshold: float = 0.55
    gallery_update_max_overlap_ratio: float = 0.25
    gallery_update_min_overall_consistency: float = 0.80
    # The design doc (ReID-Pipeline.pdf, Bước 5) specifies a single matching
    # threshold. The historical soft_match / eager_soft_match paths added in
    # earlier iterations were a backdoor that accepted matches BELOW the
    # primary similarity_threshold, which silently re-introduced cross-person
    # contamination after similarity_threshold was tightened. Align both with
    # similarity_threshold so the soft paths cannot accept what the primary
    # gallery search has rejected.
    soft_match_threshold: float = 0.73
    eager_soft_match_threshold: float = 0.73
    # Visibility-aware match threshold: tracklets with v_avg < low_visibility_threshold
    # must clear low_visibility_match_threshold (instead of similarity_threshold)
    # against an existing person before matching. Prevents partial-body / boundary
    # crops from being wrongly absorbed into existing identities at sim 0.60-0.70.
    low_visibility_threshold: float = 0.65
    low_visibility_match_threshold: float = 0.75
    # Reject a tracklet whose selected embeddings disagree with each other
    # below this threshold. Prevents mixed-identity tracklets (e.g., from
    # spatially-close untracked-detection clusters) from matching or minting,
    # which would otherwise pollute a confirmed person's gallery & snapshots.
    match_consistency_threshold: float = 0.55
    # Multi-person/occluded crops are useful evidence for recovery, but they
    # must not become canonical identity evidence.
    person_snapshot_max_overlap_ratio: float = 0.35
    current_identity_min_score: float = 0.50
    # Current track continuity is useful under occlusion, but it must lose when
    # the current person's appearance is weak and another gallery identity is
    # clearly stronger. This prevents ByteTrack ID-swap fragments from polluting
    # the previously-bound person's snapshots.
    current_identity_switch_min_score: float = 0.78
    current_identity_switch_min_margin: float = 0.18
    current_identity_switch_max_current_score: float = 0.70
    max_person_identities: int = 0
    capped_identity_soft_match_threshold: float = 0.57
    # Defer high-quality but appearance-ambiguous re-entry fragments instead
    # of minting a confirmed person. This keeps occlusion evidence visible
    # without polluting the confirmed identity set with singleton duplicates.
    # Must be < soft_match_threshold so the gate actually fires — setting it
    # equal to the match threshold disables the noise band.
    near_gallery_defer_threshold: float = 0.60
    # Narrow occlusion-only bridge for near-gallery deferred fragments. These
    # tracklets are attached as provisional sightings, never canonical/gallery
    # updates, so occluded evidence can be shown without teaching the identity
    # model from partial or boundary crops.
    occlusion_provisional_match_enabled: bool = True
    occlusion_provisional_match_threshold: float = 0.60
    occlusion_provisional_min_margin: float = 0.03
    occlusion_provisional_short_reentry_enabled: bool = True
    occlusion_provisional_reentry_min_similarity: float = 0.58
    occlusion_provisional_reentry_max_entries: int = 8
    occlusion_provisional_reentry_max_gap_frames: int = 180
    occlusion_provisional_reentry_max_center_distance_ratio: float = 2.0

    track_high_thresh: float = 0.7
    track_low_thresh: float = 0.35
    match_thresh: float = 0.3
    new_track_thresh: float = 0.82
    track_buffer: int = 90
    fuse_score: bool = True

    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "reid_production"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    person_id_seq_key: str = "reid:seq:person_id"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minio"
    minio_secret_key: str = "minio123"

    # Gender voting
    gender_person_threshold: float = 0.7
    # Gender classifier confidence can drop under occlusion/side-view. Two
    # consecutive same-label tracklets are a stronger signal than one high
    # confidence early crop, so gender gets a lower flip threshold than clothing
    # attributes while non-gender tasks keep the conservative default below.
    gender_flip_threshold: float = 0.65
    attribute_flip_threshold: float = 0.85
    glasses_best_frame_override_threshold: float = 0.6
    attribute_crop_top_padding_ratio: float = 0.22
    attribute_crop_side_padding_ratio: float = 0.08
    attribute_crop_bottom_padding_ratio: float = 0.04
    blocked_match_score_threshold: float = 0.9
    duplicate_track_iou_threshold: float = 0.5
    cooccurrence_guard_enabled: bool = True
    cooccurrence_guard_min_shared_frames: int = 1
    cooccurrence_guard_max_iou: float = 0.15
    cooccurrence_guard_min_center_distance_ratio: float = 0.55
    attribute_conflict_guard_enabled: bool = True
    attribute_conflict_person_confidence: float = 0.60
    attribute_conflict_tracklet_confidence: float = 0.88
    attribute_conflict_person_min_support: int = 1
    pretrack_static_filter_enabled: bool = False
    pretrack_static_filter_min_frames: int = 4
    pretrack_static_filter_max_width_px: float = 130.0
    pretrack_static_filter_max_height_px: float = 260.0
    pretrack_static_filter_max_center_drift_px: float = 6.0
    static_artifact_filter_enabled: bool = True
    static_artifact_max_mean_width_px: float = 130.0
    static_artifact_max_mean_height_px: float = 260.0
    # True static false positives usually produce only tiny bbox jitter. Keep
    # this filter narrow so standing or boundary-truncated people remain valid
    # occlusion evidence rather than being suppressed as background artifacts.
    static_artifact_max_path_displacement_ratio: float = 0.05
    static_artifact_max_endpoint_displacement_ratio: float = 0.02
    static_artifact_min_bbox_stability: float = 0.97
    static_artifact_min_position_stability: float = 0.97
    static_artifact_min_entries: int = 6
    # PDF Bước 1 lists cut_off (bbox chạm biên frame) as an OCCLUSION signal,
    # not a staticness one. A person standing partially out of frame produces
    # small bbox + low motion → the four AND-conditions above would suppress
    # them. Short-circuit suppression when at least this fraction of frames
    # had bbox touching any frame edge (≤2 px from boundary).
    static_artifact_boundary_contact_skip: float = 0.3
    recent_match_guard_enabled: bool = True
    recent_match_guard_seconds: float = 4.0
    recent_match_guard_min_iou: float = 0.1
    recent_match_guard_max_center_distance_ratio: float = 0.9
    gallery_update_anchor_min_score: float = 0.50
    gallery_consensus_weight: float = 0.5
    duplicate_merge_enabled: bool = True
    duplicate_merge_min_score: float = 0.90
    duplicate_merge_weak_max_tracklets: int = 2
    duplicate_merge_singleton_min_score: float = 0.90
    duplicate_merge_singleton_min_target_tracklets: int = 3
    duplicate_merge_min_margin: float = 0.10
    # Occlusion/edge re-entry often produces two short identities separated by
    # only a few frames. Allow that narrow temporal-continuity case below the
    # normal high-confidence duplicate threshold, while keeping cooccurrence and
    # attribute guards active.
    duplicate_merge_temporal_continuity_enabled: bool = True
    duplicate_merge_temporal_continuity_min_score: float = 0.85
    duplicate_merge_temporal_continuity_max_gap_frames: int = 15
    duplicate_merge_adjacent_fragment_enabled: bool = True
    duplicate_merge_adjacent_fragment_min_score: float = 0.70
    duplicate_merge_adjacent_fragment_max_gap_frames: int = 3
    duplicate_merge_occlusion_reentry_enabled: bool = True
    duplicate_merge_occlusion_reentry_min_score: float = 0.58
    duplicate_merge_occlusion_reentry_max_gap_frames: int = 180
    duplicate_merge_occlusion_reentry_max_center_distance_ratio: float = 2.0
    duplicate_merge_same_gender_singleton_enabled: bool = True
    duplicate_merge_same_gender_singleton_min_score: float = 0.80
    duplicate_merge_same_gender_singleton_gender_confidence: float = 0.80
    duplicate_merge_singleton_unknown_attr_min_score: float = 0.88
    # When two persons' canonical galleries match above this cosine threshold,
    # treat them as the same physical human regardless of attribute conflict.
    # Calibrated above typical "different people" cosine band (~0.65) so it
    # never merges distinct humans, but below the "same person across pose"
    # range so a wrong attribute prediction can no longer permanently fragment
    # one identity. Video-agnostic; depends only on the embedding model.
    duplicate_merge_attr_override_threshold: float = 0.80
    # Same principle for the cooccurrence check, but at a higher threshold
    # because cooccurrence is a stronger signal (two persons appearing in
    # overlapping frame ranges). Set above 0.80 so we don't merge two physically
    # distinct people who happen to be in the same scene; below 0.90 so a
    # same-person identity-split (where the same physical person's detections
    # got attributed to two IDs in overlapping frames) can still be repaired.
    duplicate_merge_cooccurrence_override_threshold: float = 0.85
    # Soft override: at moderate similarity (0.75-0.85), attribute conflict is
    # itself evidence of identity-split (attribute model misclassified one side
    # of the same physical person), so we override BOTH guards together.
    # Two genuinely-different-gender people who happen to look similar rarely
    # score this high in embedding cosine — high-sim + attr-flip is the
    # identity-split signature. Video-agnostic.
    duplicate_merge_soft_split_override_threshold: float = 0.70
    duplicate_merge_soft_split_max_weak_tracklets: int = 8
    duplicate_merge_soft_split_max_center_distance_ratio: float = 0.35
    duplicate_merge_soft_split_duplicate_iou_threshold: float = 0.45
    duplicate_merge_soft_split_duplicate_box_multitrack_min_score: float = 0.58
    duplicate_merge_soft_split_spatial_only_min_score: float = 0.50
    duplicate_merge_soft_split_spatial_only_multitrack_min_score: float = 0.60
    duplicate_merge_soft_split_spatial_only_max_center_distance_ratio: float = 0.30
    duplicate_merge_overlap_spatial_duplicate_enabled: bool = True
    duplicate_merge_overlap_spatial_duplicate_min_score: float = 0.58
    duplicate_merge_overlap_spatial_duplicate_max_gap_frames: int = 4
    duplicate_merge_overlap_spatial_duplicate_max_tracklets: int = 24
    duplicate_merge_overlap_spatial_duplicate_max_center_distance_ratio: float = 0.08
    duplicate_merge_overlap_spatial_duplicate_max_size_ratio: float = 1.25
    duplicate_merge_overlap_spatial_duplicate_max_area_ratio: float = 1.60
    duplicate_merge_trajectory_reentry_enabled: bool = True
    duplicate_merge_trajectory_reentry_min_score: float = 0.60
    duplicate_merge_trajectory_reentry_max_gap_frames: int = 240
    duplicate_merge_trajectory_reentry_max_tracklets: int = 24
    duplicate_merge_trajectory_reentry_max_center_distance_ratio: float = 0.06
    duplicate_merge_trajectory_reentry_max_size_ratio: float = 1.30
    duplicate_merge_trajectory_reentry_max_area_ratio: float = 2.00
    duplicate_merge_singleton_spatial_continuation_min_score: float = 0.30
    duplicate_merge_singleton_spatial_continuation_max_gap_frames: int = 15
    duplicate_merge_singleton_spatial_continuation_max_center_distance_ratio: float = 0.30
    duplicate_merge_singleton_spatial_continuation_max_size_ratio: float = 1.80
    duplicate_merge_singleton_spatial_continuation_max_area_ratio: float = 2.20
    duplicate_merge_adjacent_tight_continuation_enabled: bool = True
    duplicate_merge_adjacent_tight_continuation_min_score: float = 0.50
    duplicate_merge_adjacent_tight_continuation_max_gap_frames: int = 4
    duplicate_merge_adjacent_tight_continuation_max_tracklets: int = 8
    duplicate_merge_adjacent_tight_continuation_max_center_distance_ratio: float = 0.06
    duplicate_merge_adjacent_tight_continuation_max_size_ratio: float = 1.10
    duplicate_merge_adjacent_tight_continuation_max_area_ratio: float = 1.15
    duplicate_merge_boundary_weak_continuation_enabled: bool = True
    duplicate_merge_boundary_weak_continuation_min_score: float = 0.50
    duplicate_merge_boundary_weak_continuation_max_gap_frames: int = 12
    duplicate_merge_boundary_weak_continuation_max_weak_tracklets: int = 2
    duplicate_merge_boundary_weak_continuation_max_supported_tracklets: int = 8
    duplicate_merge_boundary_weak_continuation_min_center_distance_ratio: float = 0.10
    duplicate_merge_boundary_weak_continuation_max_center_distance_ratio: float = 0.32
    duplicate_merge_boundary_weak_continuation_max_size_ratio: float = 1.80
    duplicate_merge_boundary_weak_continuation_max_area_ratio: float = 2.30
    duplicate_merge_boundary_weak_continuation_max_bottom_delta_ratio: float = 0.06
    duplicate_merge_boundary_duplicate_min_score: float = 0.68
    duplicate_merge_boundary_duplicate_min_iou: float = 0.10
    duplicate_merge_boundary_duplicate_max_center_distance_ratio: float = 0.45
    duplicate_merge_ultra_continuity_min_score: float = 0.50
    duplicate_merge_ultra_continuity_max_gap_frames: int = 6
    duplicate_merge_ultra_continuity_max_center_distance_ratio: float = 0.12
    duplicate_merge_ultra_continuity_max_weak_tracklets: int = 2
    duplicate_merge_ultra_continuity_max_supported_tracklets: int = 8
    duplicate_merge_tight_spatial_reentry_enabled: bool = True
    duplicate_merge_tight_spatial_reentry_min_score: float = 0.50
    duplicate_merge_tight_spatial_reentry_max_gap_frames: int = 6
    duplicate_merge_tight_spatial_reentry_max_center_distance_ratio: float = 0.12
    duplicate_merge_tight_spatial_reentry_max_size_ratio: float = 1.15
    duplicate_merge_tight_spatial_reentry_max_area_ratio: float = 1.25
    duplicate_merge_tight_spatial_reentry_max_weak_tracklets: int = 2
    duplicate_merge_same_frame_established_duplicate_enabled: bool = True
    duplicate_merge_same_frame_established_duplicate_min_score: float = 0.50
    duplicate_merge_same_frame_established_duplicate_min_iou: float = 0.75
    duplicate_merge_same_frame_established_duplicate_max_center_distance_ratio: float = 0.05
    duplicate_merge_same_frame_established_duplicate_max_size_ratio: float = 1.15
    duplicate_merge_same_frame_established_duplicate_max_area_ratio: float = 1.25
    duplicate_merge_reentry_bridge_enabled: bool = True
    duplicate_merge_reentry_bridge_min_score: float = 0.535
    duplicate_merge_reentry_bridge_max_tracklets: int = 4
    duplicate_merge_reentry_bridge_max_supported_tracklets: int = 8
    duplicate_merge_reentry_bridge_min_gap_frames: int = 30
    duplicate_merge_reentry_bridge_max_gap_frames: int = 180
    duplicate_merge_reentry_bridge_max_center_distance_ratio: float = 0.85
    duplicate_merge_reentry_bridge_gender_confidence: float = 0.70
    duplicate_merge_reentry_bridge_min_attr_matches: int = 2
    duplicate_merge_reentry_bridge_supported_min_score: float = 0.70
    duplicate_merge_reentry_bridge_supported_min_margin: float = 0.12
    duplicate_merge_supported_spatial_reentry_enabled: bool = True
    duplicate_merge_supported_spatial_reentry_min_score: float = 0.53
    duplicate_merge_supported_spatial_reentry_max_tracklets: int = 8
    duplicate_merge_supported_spatial_reentry_min_gap_frames: int = 24
    duplicate_merge_supported_spatial_reentry_max_gap_frames: int = 90
    duplicate_merge_supported_spatial_reentry_max_center_distance_ratio: float = 0.18
    duplicate_merge_supported_spatial_reentry_max_size_ratio: float = 1.20
    duplicate_merge_supported_spatial_reentry_max_area_ratio: float = 1.80
    # Clothing-only long re-entry is useful as a candidate signal, but it is
    # not identity-safe enough to hard-merge into a confirmed person. Keep it
    # off by default so weak occlusion fragments stay tentative/evidence until
    # stronger appearance or geometry supports them.
    duplicate_merge_clothing_reentry_enabled: bool = False
    duplicate_merge_clothing_reentry_min_score: float = 0.515
    duplicate_merge_clothing_reentry_max_weak_tracklets: int = 2
    duplicate_merge_clothing_reentry_max_supported_tracklets: int = 8
    duplicate_merge_clothing_reentry_min_gap_frames: int = 30
    duplicate_merge_clothing_reentry_max_gap_frames: int = 240
    duplicate_merge_clothing_reentry_min_attr_matches: int = 3
    duplicate_merge_clothing_reentry_attr_confidence: float = 0.70
    duplicate_merge_weak_to_supported_guard_enabled: bool = True
    duplicate_merge_weak_to_supported_min_target_tracklets: int = 5
    duplicate_merge_weak_to_supported_max_target_tracklets: int = 8
    duplicate_merge_weak_to_supported_min_score: float = 0.78
    duplicate_merge_weak_to_supported_min_margin: float = 0.08
    duplicate_merge_weak_to_supported_strong_score: float = 0.89
    duplicate_merge_weak_to_supported_strong_margin: float = 0.18
    duplicate_merge_occlusion_spatial_rejoin_enabled: bool = False
    duplicate_merge_occlusion_spatial_rejoin_min_score: float = 0.53
    duplicate_merge_occlusion_spatial_rejoin_strong_min_score: float = 0.59
    duplicate_merge_occlusion_spatial_rejoin_max_gap_frames: int = 180
    duplicate_merge_occlusion_spatial_rejoin_max_center_distance_ratio: float = 0.50
    duplicate_merge_occlusion_spatial_rejoin_tight_center_distance_ratio: float = 0.42
    duplicate_merge_occlusion_spatial_rejoin_max_size_ratio: float = 1.55
    duplicate_merge_occlusion_spatial_rejoin_max_area_ratio: float = 2.10
    duplicate_merge_occlusion_spatial_rejoin_tight_size_ratio: float = 1.10
    duplicate_merge_occlusion_spatial_rejoin_tight_area_ratio: float = 2.00
    duplicate_merge_spatial_continuation_enabled: bool = False
    duplicate_merge_spatial_continuation_min_score: float = 0.20
    duplicate_merge_spatial_continuation_max_gap_frames: int = 60
    duplicate_merge_spatial_continuation_max_center_distance_ratio: float = 0.30
    # Per-sighting confidence required for a gender label to count when
    # checking cross-gender disagreement at merge time. Attribute disagreement
    # is supporting evidence only; high appearance confidence can still override
    # noisy attributes through the duplicate-merge policy above.
    gender_block_sighting_confidence: float = 0.80

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="WORKER_",
        extra="ignore",
    )


settings = Settings()
