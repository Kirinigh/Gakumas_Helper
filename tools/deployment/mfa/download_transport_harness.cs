// Offline fixture: production bodies are inserted by test_download_transport.py.
// UI, environment/proxy selection and extraction are instrumented seams; the
// HTTP transport, retry delays, download/verification gate and queue are real.
using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

public static class Program
{
    private static readonly ConcurrentQueue<(string Name, Func<Task> Action)> Queue = new();
    private static readonly SemaphoreSlim _queueLock = new(1, 1);
    private static HttpClient CreateHttpClientWithProxy() => new(new SocketsHttpHandler { UseProxy = false });
    private static void SetProgress(ProgressBar? progress, double value) { if (progress != null) progress.Values.Add(value); }
    private static void SetDownloadInfo(TextBlock? size, TextBlock? speed, long read, long total, long perSecond) { }
    private static void SetStatusText(TextBlock? text, TextBlock? speed, string message) { }
    private static void Dismiss(object? toast) { }
    private static string GetLocalPackageExtractDirectory(string path) => path + "_extracted";

    // PRODUCTION_METHODS

    private static async Task ResourceGate(string downloadUrl, string tempZipFilePath, string sha256, ProgressBar progress)
    {
        var isLocalPackage = false;
        string? localPackagePath = null;
        TextBlock? textBlock = null, downloadSpeedTextBlock = null;
        object? sukiToast = null;
        var tempPath = Path.GetDirectoryName(tempZipFilePath)!;
        var keepArtifacts = false;
        var latestVersion = "test";
        var debugSessionId = "test";
        // PRODUCTION_RESOURCE_GATE
    }

    private static void TaskCleanup()
    {
        // PRODUCTION_TASK_CLEANUP
    }

    private static void Require(bool value, string reason)
    {
        if (!value) throw new Exception(reason);
    }

    public static async Task Main(string[] args)
    {
        Directory.CreateDirectory(args[0]);
        var payload = Enumerable.Range(0, 32768).Select(i => (byte)(i % 251)).ToArray();
        var scenarios = new[] {
            ("complete", new[] { "ok" }, true, 1),
            ("disconnect_then_success", new[] { "short", "ok" }, true, 2),
            ("http_failure_then_success", new[] { "http500", "ok" }, true, 2),
            ("disconnect_exhausted", new[] { "short" }, false, 3),
            ("http_failure_exhausted", new[] { "http500" }, false, 3),
            ("empty_exhausted", new[] { "empty" }, false, 3),
            ("close_delimited_complete", new[] { "no_length" }, true, 1),
            ("chunked_disconnect_then_success", new[] { "chunk_short", "ok" }, true, 2),
            ("content_disposition_retry", new[] { "renamed_short", "renamed_ok" }, true, 2),
        };
        int passed = 0;
        foreach (var (name, modes, expected, requests) in scenarios)
        {
            await using var server = new LoopbackServer(modes, payload);
            var file = Path.Combine(args[0], name + ".zip");
            var progress = new ProgressBar();
            var (success, actualPath) = await DownloadWithRetry(server.Url, file, progress, 3);
            Require(success == expected, name + ": success status");
            Require(server.Requests == requests, name + ": bounded request count");
            if (success)
            {
                Require(File.ReadAllBytes(actualPath).SequenceEqual(payload), name + ": byte identity");
                Require(progress.Values.Last() == 100, name + ": final progress");
            }
            else
            {
                Require(!File.Exists(actualPath), name + ": partial download removed");
                Require(!progress.Values.Contains(100), name + ": never show complete progress");
            }
            Console.WriteLine($"PASS {name}: requests={server.Requests}");
            passed++;
        }

        foreach (var mode in new[] { "short", "http500", "wrong_digest", "ok" })
        {
            await using var server = new LoopbackServer(new[] { mode == "wrong_digest" ? "ok" : mode }, payload);
            LoggerHelper.Messages.Clear();
            UniversalExtractor.Calls = 0;
            Instances.RootViewModel.Updating = true;
            var expectedHash = Convert.ToHexString(SHA256.HashData(mode == "wrong_digest" ? new byte[] { 1 } : payload));
            Queue.Enqueue(("resource", () => ResourceGate(server.Url, Path.Combine(args[0], "gate_" + mode + ".zip"), expectedHash, new ProgressBar())));
            Exception? error = null;
            try { await ExecuteTasksAsync(); }
            catch (Exception ex) { error = ex; }
            Require(!Instances.RootViewModel.Updating, mode + ": clear updating state");
            if (mode == "ok")
            {
                Require(error == null && UniversalExtractor.Calls == 1, "complete gate reaches extraction once");
                Require(LoggerHelper.Messages.Any(m => m.Contains("更新任务完成")), "completed task message");
            }
            else
            {
                Require(error != null, mode + ": propagate failure to TaskManager");
                Require(UniversalExtractor.Calls == 0, mode + ": no extraction/application");
                Require(!LoggerHelper.Messages.Any(m => m.Contains("更新任务完成")), mode + ": no task success message");
                if (mode != "wrong_digest")
                    Require(!LoggerHelper.Messages.Any(m => m.Contains("已下载更新包")), mode + ": no downloaded message");
            }
            Console.WriteLine($"PASS resource_gate_{mode}");
            passed++;
        }

        AppPaths.Root = Path.Combine(args[0], "task_cleanup");
        Directory.CreateDirectory(AppPaths.TempResourceDirectory);
        var activeDownload = Path.Combine(AppPaths.TempResourceDirectory, "resource.zip");
        using (var held = new FileStream(activeDownload, FileMode.Create, FileAccess.Write, FileShare.None))
        {
            TaskCleanup();
            held.Write(payload);
        }
        Require(File.ReadAllBytes(activeDownload).SequenceEqual(payload), "task cleanup preserves active download");
        Console.WriteLine("PASS task_cleanup_preserves_active_download");
        passed++;
        Console.WriteLine($"TOTAL {passed} PASS");
    }
}

public sealed class LoopbackServer : IAsyncDisposable
{
    private readonly TcpListener listener = new(IPAddress.Loopback, 0);
    private readonly string[] modes;
    private readonly byte[] body;
    private readonly Task serving;
    private int requests;
    public int Requests => Volatile.Read(ref requests);
    public string Url { get; }
    public LoopbackServer(string[] modes, byte[] body)
    {
        this.modes = modes; this.body = body;
        listener.Start();
        Url = $"http://127.0.0.1:{((IPEndPoint)listener.LocalEndpoint).Port}/resource.zip";
        serving = Serve();
    }
    private async Task Serve()
    {
        try
        {
            while (true)
            {
                using var client = await listener.AcceptTcpClientAsync();
                await using var stream = client.GetStream();
                using var reader = new StreamReader(stream, Encoding.ASCII, false, 1024, true);
                while (!string.IsNullOrEmpty(await reader.ReadLineAsync())) { }
                var index = Interlocked.Increment(ref requests) - 1;
                var mode = modes[Math.Min(index, modes.Length - 1)];
                var status = mode == "http500" ? "500 Internal Server Error" : "200 OK";
                var length = mode is "http500" or "empty" ? 0 : body.Length;
                var header = $"HTTP/1.1 {status}\r\nConnection: close\r\n";
                if (mode.StartsWith("renamed")) header += "Content-Disposition: attachment; filename=resource.tar\r\n";
                if (mode == "chunk_short") header += "Transfer-Encoding: chunked\r\n";
                else if (mode != "no_length") header += $"Content-Length: {length}\r\n";
                await stream.WriteAsync(Encoding.ASCII.GetBytes(header + "\r\n"));
                if (mode == "chunk_short")
                    await stream.WriteAsync(Encoding.ASCII.GetBytes(body.Length.ToString("X") + "\r\n"));
                var count = mode is "short" or "chunk_short" or "renamed_short" ? body.Length / 2 : length;
                if (count > 0) await stream.WriteAsync(body.AsMemory(0, count));
                await stream.FlushAsync();
            }
        }
        catch (SocketException) { }
        catch (ObjectDisposedException) { }
    }
    public async ValueTask DisposeAsync() { listener.Stop(); await serving; }
}

public sealed class ProgressBar { public List<double> Values { get; } = new(); }
public sealed class TextBlock { }
public static class LoggerHelper
{
    public static List<string> Messages { get; } = new();
    public static void Info(string message) => Messages.Add(message);
    public static void Warning(string message) => Messages.Add(message);
    public static void Error(string message, Exception error) => Messages.Add(message);
}
public static class DispatcherHelper { public static void PostOnMainThread(Action action) => action(); }
public static class Instances
{
    public static RootModel RootViewModel { get; } = new();
    public static Settings VersionUpdateSettingsUserControlModel { get; } = new();
    public static Tabs InstanceTabBarViewModel { get; } = new();
}
public sealed class RootModel { public bool Updating; public void SetUpdating(bool value) => Updating = value; }
public sealed class Settings { public string GitHubToken => ""; }
public sealed class Tabs { public Tab? ActiveTab => null; }
public sealed class Tab { public TaskQueueView TaskQueueViewModel { get; } = new(); }
public sealed class TaskQueueView { public void OutputDownloadProgress(long read, long total, int speed, double time) { } }
public enum LangKeys { Downloading, Warning, DownloadFailed, Extracting, Verifying, HashVerificationFailed, ApplyingUpdate }
public static class Localization { public static string ToLocalization(this LangKeys key) => key.ToString(); }
public static class ToastHelper { public static void Warn(string title, string message) { } }
public static class UniversalExtractor { public static int Calls; public static void Extract(string source, string target) => Calls++; }
public static class AppPaths
{
    public static string Root = "";
    public static string TempMfaDirectory => Path.Combine(Root, "temp_mfa");
    public static string TempMaaFwDirectory => Path.Combine(Root, "temp_maafw");
    public static string TempResourceDirectory => Path.Combine(Root, "temp_res");
}
