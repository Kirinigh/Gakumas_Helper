"""Shared HIF business recovery; no CLI, startup, or input authorization."""


def diagnostic_depth_reached(ancestors):
    """Keep the standalone replay limit outside the automatic battle callback."""
    if len(ancestors) < 60:
        return False
    from ..hif_live_recovery import automatic_host_active

    return not automatic_host_active()


RUNTIME_MODULES = ('native_card_evidence', 'native_choice_resume', 'native_fast_cache', 'native_hand_support', 'native_handoff_resume', 'native_history_navigation', 'native_inprocess_resume', 'native_manual_choice_restart', 'native_opening_resume', 'native_opening_root', 'native_pending_components', 'native_pending_restart', 'native_preinput_history', 'native_restart_source', 'native_rule_process_recovery', 'native_rule_progress_resume', 'native_search_cache', 'native_search_window', 'native_selected_resume', 'native_selected_support', 'native_settled_restart', 'native_settled_restart_components', 'native_settlement_resume', 'native_single_plus', 'native_support_resume', 'native_visible_history')
