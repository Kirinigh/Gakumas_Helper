using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

enum MaaControllerTypes { Adb, Win32, Gamepad, PlayCover }
interface IBrush {}
static class LangKeys { public const string Emulator="Emulator", Window="Window", TabPlayCover="PlayCover", ConnectingTo="Connecting", Tip="Tip", TryToStartGame="Start", ConnectFailed="Failed"; }
static class ConfigurationKeys { public const string BeforeTask="before", AutoDetectOnConnectionFailed="detect", RetryOnDisconnectedWin32="retry"; }
static class Strings { public static string ToLocalization(this string s)=>s; public static string ToLocalizationFormatted(this string s,bool b,string arg)=>s; }
static class LoggerHelper { public static void Warning(string s){} }
static class ToastHelper { public static void Info(string a,string b){} }
static class TelemetryService { public static void RecordTaskFailure(string a,string b,string c,Exception e){} }
static class MFATask { public enum MFATaskStatus { STOPPED } }
class Settings { public Dictionary<string,object> Values=new(); public T GetValue<T>(string key,T fallback)=>Values.TryGetValue(key,out var v)?(T)v:fallback; }
class DeviceConfig { public string AdbSerial="device"; public nint HWnd=1; }
class ConfigState { public DeviceConfig AdbDevice=new(), DesktopWindow=new(); }
class ControllerState { public bool IsConnected; }
class TaskerState { public bool IsInitialized; public ControllerState Controller=new(); }
class VM {
 public MaaControllerTypes CurrentController=MaaControllerTypes.Win32; public bool IsConnected; public object? CurrentDevice=new(); public int Refreshes; public Action? Refresh;
 public void SetConnected(bool value)=>IsConnected=value;
 public void SetAdbRecoverySelectionLock(bool value){}
 public void SyncCurrentAdbSelectionToActiveConfig(){}
 public void TryReadAdbDeviceFromConfig(bool a,bool b,bool c,bool d,bool e=false){Refreshes++;Refresh?.Invoke();}
}
partial class Harness {
 VM ViewModel=new(); ConfigState Config=new(); Settings InstanceConfiguration=new(); TaskerState? MaaTasker;
 SemaphoreSlim _connectionGate=new(1,1); int _isConnecting; bool _suppressConnectionAttemptErrorToast;
 string InstanceId="test"; const string ConnectionFailedAfterAllRetriesMessage="Connection failed after all retries";
 Queue<bool> NativeResults=new(); int NativeCalls, AdbRetries, Started; List<string> Logs=new();
 bool ShouldStrictMatchSavedAdbTarget()=>false;
 Task EnsureAdbTargetReadyAsync(CancellationToken token,bool show,bool delay)=>Task.CompletedTask;
 Task<bool> HandleAdbConnectionAsync(CancellationToken token,bool show){AdbRetries++;return Task.FromResult(false);}
 void HandleConnectionFailureAsync(MaaControllerTypes t,CancellationToken c){}
 void AddLog(string message,IBrush? brush=null)=>Logs.Add(message);
 void AddLogByKey(string a,IBrush? b,bool c,bool d,string e){}
 void Stop(MFATask.MFATaskStatus status){}
 Task StartSoftware(){Started++;return Task.CompletedTask;}
 static bool IsLiveConnectionWindow(nint h)=>h==1;
 Task<(TaskerState?,bool,bool)> GetTaskerAndBoolAsync(CancellationToken token){
  NativeCalls++;bool ok=NativeResults.Count>0&&NativeResults.Dequeue();
  MaaTasker=new(){IsInitialized=ok,Controller=new(){IsConnected=ok}};
  return Task.FromResult<(TaskerState?,bool,bool)>((MaaTasker,false,true));
 }
 static void Check(bool condition,string label){if(!condition)throw new Exception(label);}
 static async Task Fails(Harness h){try{await h.HandleDeviceConnectionAsync(CancellationToken.None);throw new Exception("unexpected success");}catch(InvalidOperationException){}}
 static async Task Main(){
  foreach(var message in new[]{"Node.Recognition.Starting","Tasker.Task.Starting","", "Node.Recognition.Succeeded.extra"})
   Check(!CanReadRecognitionResult(message),"premature result "+message);
  foreach(var message in new[]{"Node.Recognition.Succeeded","Node.Recognition.Failed","Node.Action.Starting","Node.Action.Succeeded","Node.Action.Failed"})
   Check(CanReadRecognitionResult(message),"completed result "+message);
  var h=new Harness();h.ViewModel.CurrentController=MaaControllerTypes.Adb;h.Config.AdbDevice.AdbSerial=" ";await Fails(h);
  Check(h.NativeCalls==0&&h.AdbRetries==0&&h.Logs.Exists(x=>x.Contains("DMM")),"empty adb repeated native retries");
  h=new();h.Config.DesktopWindow.HWnd=0;h.ViewModel.IsConnected=true;h.MaaTasker=new(){IsInitialized=true,Controller=new(){IsConnected=true}};
  h.ViewModel.Refresh=()=>h.Config.DesktopWindow.HWnd=1;h.NativeResults.Enqueue(true);await h.HandleDeviceConnectionAsync(CancellationToken.None);
  Check(h.ViewModel.Refreshes==1&&h.NativeCalls==1&&h.ViewModel.IsConnected,"stale connected window not refreshed");
  h=new();h.Config.DesktopWindow.HWnd=2;await Fails(h);Check(h.NativeCalls==0&&!h.ViewModel.IsConnected,"invalid hwnd used");
  h=new();h.Config.DesktopWindow.HWnd=0;h.InstanceConfiguration.Values["detect"]=false;await Fails(h);Check(h.ViewModel.Refreshes==0,"auto detect opt-out ignored");
  h=new();h.InstanceConfiguration.Values["retry"]=true;h.NativeResults.Enqueue(false);h.NativeResults.Enqueue(true);
  await h.HandleDeviceConnectionAsync(CancellationToken.None);Check(h.NativeCalls==2&&h.Started==1&&h.ViewModel.IsConnected,"successful retry discarded");
  h=new();h.InstanceConfiguration.Values["retry"]=true;h.NativeResults.Enqueue(false);h.NativeResults.Enqueue(false);await Fails(h);Check(!h.ViewModel.IsConnected,"failed retry accepted");
  h=new();h.NativeResults.Enqueue(true);await h.HandleDeviceConnectionAsync(CancellationToken.None);Check(h.NativeCalls==1&&h.Started==0,"normal connect changed");
  using var cts=new CancellationTokenSource();cts.Cancel();h=new();try{await h.HandleDeviceConnectionAsync(cts.Token);throw new Exception("cancel ignored");}catch(OperationCanceledException){}Check(h.NativeCalls==0,"cancel issued native connect");
  Check(h._connectionGate.CurrentCount==1,"connection gate leaked");
  Console.WriteLine("PASS: 17 callback and actual connection control-flow checks; no native device calls");
 }
}
