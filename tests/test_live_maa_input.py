from unittest.mock import Mock, patch

import pytest

from agent.utils import maa_input


@pytest.mark.parametrize('size', [(720, 1280), (719, 1279), (480, 853), (479, 852), (481, 854), (721, 1282),
                                 (540, 960), (541, 961), (541, 962), (542, 962), (359, 638), (360, 640)])
@pytest.mark.parametrize('failure', [None, 'resize', 'medium_child', 'unknown_child'])
def test_input_checks_observed_size_and_every_actual_process(
        size, failure, actual_size=None, require_calibrated_size=True):
    user = Mock()
    kernel = Mock()
    user.GetWindowTextW.side_effect = lambda hwnd, buffer, length: setattr(buffer, 'value', 'gakumas')
    user.GetClassNameW.side_effect = lambda hwnd, buffer, length: setattr(buffer, 'value', 'UnityWndClass')

    def window_pid(hwnd, pointer):
        pointer._obj.value = 200
        return 1

    def rect(hwnd, pointer):
        pointer._obj.right, pointer._obj.bottom = (actual_size or size) if failure != 'resize' else (482, 853)
        return True

    def executable(handle, flags, buffer, length):
        buffer.value = {100: 'python.exe', 101: 'python.exe', 102: 'MaaPiCli.exe'}[handle]
        return True

    user.GetWindowThreadProcessId.side_effect = window_pid
    user.GetClientRect.side_effect = rect
    kernel.OpenProcess.side_effect = lambda access, inherit, pid: pid
    kernel.QueryFullProcessImageNameW.side_effect = executable

    def integrity(pid):
        if pid == 101 and failure == 'unknown_child':
            raise maa_input.MaaInputEndpointError('could not read child token')
        return 8192 if pid == 101 and failure == 'medium_child' else 12288

    with (patch.object(maa_input.os, 'name', 'nt'),
          patch.object(maa_input.os, 'getpid', return_value=100),
          patch.object(maa_input.ctypes, 'WinDLL', side_effect=lambda name, **kw: user if name == 'user32' else kernel),
          patch.object(maa_input, '_parent_process_ids', return_value={100: 101, 101: 102}),
          patch.object(maa_input, 'process_integrity_rid', side_effect=integrity)):
        info = {'type': 'win32', 'hwnd': 5, 'mouse_method': 32}
        if failure:
            with pytest.raises(maa_input.MaaInputEndpointError):
                maa_input.verify_maa_input_endpoints(info, expected_client_size=size,
                                                     require_calibrated_size=require_calibrated_size)
        else:
            result = maa_input.verify_maa_input_endpoints(info, expected_client_size=size,
                                                           require_calibrated_size=require_calibrated_size)
            assert {row['pid'] for row in result.values()} == {100, 101, 102, 200}
            assert result['input_child_1']['integrity_rid'] == 12288
            assert result['game']['observed_client_size'] == (actual_size or size)


@pytest.mark.parametrize('size', [(812, 1443), (1080, 1920), (2160, 3840)])
def test_normalized_capture_can_use_an_unlisted_but_stable_client_size(size):
    with pytest.raises(maa_input.MaaInputEndpointError, match='unsupported observed'):
        maa_input.verify_maa_input_endpoints({}, expected_client_size=size)
    test_input_checks_observed_size_and_every_actual_process(size, None, require_calibrated_size=False)
    test_input_checks_observed_size_and_every_actual_process(size, 'medium_child', require_calibrated_size=False)
    actual = (size[0] + 2, size[1])
    with pytest.raises(maa_input.MaaInputEndpointError, match='size changed'):
        test_input_checks_observed_size_and_every_actual_process(size, None, actual, require_calibrated_size=False)


@pytest.mark.parametrize('size', [(480, 855), (478, 853), (719, 1280), (1280, 720),
                                 (540, 962), (543, 963), (480,), (480.0, 853), [480, 853], None])
def test_unknown_client_geometry_never_reaches_windows(size):
    with patch.object(maa_input.ctypes, 'WinDLL') as windows:
        with pytest.raises(maa_input.MaaInputEndpointError, match='unsupported observed'):
            maa_input.verify_maa_input_endpoints({}, expected_client_size=size)
        windows.assert_not_called()


@pytest.mark.parametrize('width', [479, 480, 481])
@pytest.mark.parametrize('height', [852, 853, 854])
def test_single_pixel_rounding_uses_fixed_small_window_profile(width, height):
    from agent.live_decision.inventory import inventory_button_template
    assert maa_input.calibrated_live_client_size((width, height)) == (480, 853)
    assert inventory_button_template((width, height)) == inventory_button_template((480, 853))


@pytest.mark.parametrize('actual,accepted', [((479, 852), True), ((480, 853), True),
                                           ((481, 854), False), ((478, 851), False)])
def test_pixel_drift_stays_bound_to_both_profile_and_capture_reference(actual, accepted):
    if accepted:
        test_input_checks_observed_size_and_every_actual_process((479, 852), None, actual)
    else:
        with pytest.raises(maa_input.MaaInputEndpointError):
            test_input_checks_observed_size_and_every_actual_process((479, 852), None, actual)


def test_display_change_profile_keeps_actual_pixel_drift_gate():
    from agent.live_decision.inventory import inventory_button_template

    assert inventory_button_template((542, 962)) == inventory_button_template((720, 1280))
    test_input_checks_observed_size_and_every_actual_process((541, 961), None, (542, 962))
    with pytest.raises(maa_input.MaaInputEndpointError, match='size changed'):
        test_input_checks_observed_size_and_every_actual_process((540, 960), None, (542, 962))
