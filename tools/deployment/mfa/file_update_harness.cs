using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using MFAAvalonia.Helper;
using Newtonsoft.Json.Linq;

namespace MFAAvalonia.Helper
{
    static class LoggerHelper
    {
        public static void Info(string value) { }
        public static void Warning(string value) { }
        public static void Error(string value, Exception? error = null) { }
    }
    static class VersionChecker
    {
        // PRODUCTION_TRANSACTION
        // PRODUCTION_CHANGELOG
        // PRODUCTION_RELEASE
        internal static void SavePreview() => SaveRelease(new JObject { ["body"] = "downloaded preview" }, "body");
        internal static void SaveNotes() => SaveChangelog(new JObject { ["body"] = "downloaded release notes" }, "body");
    }
    static class AppPaths { public static string ResourceDirectory { get; set; } = ""; }
    static class ChangelogViewModel { public const string ChangelogFileName = "Changelog.md"; public const string ReleaseFileName = "Release.md"; }
    static class ConfigurationKeys { public const string DoNotShowChangelogAgain = "DoNotShowChangelogAgain"; }
    static class GlobalConfiguration { public static void SetValue(string key, string value) { } }
}
sealed class FixtureHandler(Dictionary<string, string> assets) : HttpMessageHandler
{
    public List<string> Requests = new();
    public Dictionary<string, int> Failures = new();
    public string? RateId;
    public string? LogPath;
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken token)
    {
        var id = request.RequestUri!.AbsolutePath.Split('/').Last();
        Requests.Add(id);
        if (LogPath != null) File.AppendAllText(LogPath, id + "\n");
        if (id == RateId)
        {
            var limited = new HttpResponseMessage((HttpStatusCode)429);
            limited.Headers.RetryAfter = new System.Net.Http.Headers.RetryConditionHeaderValue(TimeSpan.FromMinutes(2));
            return Task.FromResult(limited);
        }
        if (Failures.TryGetValue(id, out var remaining) && remaining > 0)
        {
            Failures[id]--;
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.InternalServerError));
        }
        if (!assets.TryGetValue(id, out var path)) throw new Exception("Unknown fake asset " + id);
        return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(File.ReadAllBytes(path)) });
    }
}
// Deliberately ignores cancellation: the caller must still enforce its deadline.
sealed class HeldBody : MemoryStream
{
    public readonly TaskCompletionSource<int> End = new(TaskCreationOptions.RunContinuationsAsynchronously);
    public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken token) => End.Task;
    public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken token = default) => new(End.Task);
}
sealed class Handler : HttpMessageHandler
{
    public int Calls;
    public readonly HeldBody Body = new();
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken token)
    {
        Calls++;
        return Task.FromResult(new HttpResponseMessage(Calls == 1 ? HttpStatusCode.Forbidden : HttpStatusCode.OK)
        {
            Content = Calls == 1 ? new StreamContent(Body) : new StringContent("ok")
        });
    }
}
static class Program
{
    static async Task Main(string[] args)
    {
        try { await Run(args); }
        catch (Exception error) { Console.Error.WriteLine(error); Environment.ExitCode = 1; }
    }
    static async Task Run(string[] args)
    {
        if (args[0] == "--directory-safety")
        {
            var fixtureRoot = args[1];
            var target = Path.Combine(fixtureRoot, "target");
            Directory.CreateDirectory(Path.Combine(target, "empty"));
            var source = Path.Combine(fixtureRoot, "source");
            File.WriteAllText(source, "new");
            File.WriteAllText(Path.Combine(target, "keep.txt"), "user content");
            using (var transaction = new VersionChecker.UpdateFileTransaction(fixtureRoot))
            {
                try { transaction.ReplaceFile(source, target, CancellationToken.None); throw new Exception("Expected nonempty directory refusal"); }
                catch (IOException) { }
                if (!transaction.RollbackChecked()) throw new Exception("Guard rollback failed");
            }
            if (File.ReadAllText(Path.Combine(target, "keep.txt")) != "user content" || !Directory.Exists(Path.Combine(target, "empty")))
                throw new Exception("Unknown contents changed");
            var emptyTarget = Path.Combine(fixtureRoot, "empty-target");
            Directory.CreateDirectory(Path.Combine(emptyTarget, "nested"));
            using (var transaction = new VersionChecker.UpdateFileTransaction(fixtureRoot))
            {
                try { transaction.ReplaceFile(source + ".missing", emptyTarget, CancellationToken.None); throw new Exception("Expected copy failure"); }
                catch (IOException) { }
                if (!transaction.RollbackChecked()) throw new Exception("Empty directory restoration failed");
            }
            if (!Directory.Exists(Path.Combine(emptyTarget, "nested"))) throw new Exception("Original empty directory lost");
            Console.WriteLine("unknown contents preserved and empty directories restored PASS");
            return;
        }
        if (args[0] == "--network-timeout")
        {
            var timeoutHandler = new Handler();
            using var timeoutClient = new HttpClient(timeoutHandler) { Timeout = TimeSpan.FromMilliseconds(100) };
            using var firstRequest = new HttpRequestMessage(HttpMethod.Get, "https://example.invalid/asset");
            try
            {
                using var unexpected = await GitHubApiRequests.SendAsync(timeoutClient, firstRequest, HttpCompletionOption.ResponseHeadersRead).WaitAsync(TimeSpan.FromSeconds(3));
                throw new Exception("Expected error-body timeout");
            }
            catch (OperationCanceledException) { }
            finally { timeoutHandler.Body.End.TrySetResult(0); }
            using var secondRequest = new HttpRequestMessage(HttpMethod.Get, "https://example.invalid/next");
            using var recovered = await GitHubApiRequests.SendAsync(timeoutClient, secondRequest, HttpCompletionOption.ResponseHeadersRead).WaitAsync(TimeSpan.FromSeconds(3));
            if (recovered.StatusCode != HttpStatusCode.OK || timeoutHandler.Calls != 2) throw new Exception("Request gate remained blocked");
            Console.WriteLine("error-body deadline and gate recovery PASS");
            return;
        }
        if (args.Length == 2 && args[0] == "--launch-plan")
        {
            var prepared = JsonSerializer.Deserialize<DerivedFileUpdate.Plan>(File.ReadAllText(args[1]))!;
            await DerivedFileUpdate.LaunchWorker(prepared, "", "");
            return;
        }
        var input = JsonNode.Parse(File.ReadAllText(args[0]))!;
        var root = (string)input["root"]!;
        if ((bool?)input["save_changelog"] == true)
        {
            AppPaths.ResourceDirectory = Path.Combine(root, "resource");
            VersionChecker.SaveNotes();
            if (!File.Exists(Path.Combine(AppPaths.ResourceDirectory, "Changelog.md")))
                throw new Exception("Production SaveChangelog did not create the downloaded notes");
        }
        if ((bool?)input["save_release"] == true)
        {
            AppPaths.ResourceDirectory = Path.Combine(root, "resource");
            VersionChecker.SavePreview();
            if (!File.Exists(Path.Combine(AppPaths.ResourceDirectory, "Release.md")))
                throw new Exception("Production SaveRelease did not create preview");
        }
        var work = (string)input["work"]!;
        Directory.CreateDirectory(work);
        var assets = input["assets"]!.Deserialize<Dictionary<string, string>>()!;
        using var handler = new FixtureHandler(assets);
        handler.Failures = input["failures"]?.Deserialize<Dictionary<string, int>>() ?? new();
        handler.RateId = (string?)input["rate_id"];
        handler.LogPath = Path.Combine(work, "requests.log");
        GitHubApiRequests.SharedRoot = root;
        using var client = new HttpClient(handler);
        client.DefaultRequestHeaders.UserAgent.ParseAdd("fixture");
        var catalog = input["catalog"]!.AsArray().Select(r => r!.AsObject()).ToList();
        var progress = new List<(string Stage, long Received, long Total)>();
        var plan = await DerivedFileUpdate.Prepare(root, work, (string)input["from"]!, (string)input["to"]!, catalog, client,
            (stage, received, total) => progress.Add((stage, received, total)));
        if (!progress.Any(p => p.Received == 0 && p.Total > 0)
            || !progress.Any(p => p.Received > 0 && p.Received <= p.Total)
            || !progress.Any(p => p.Stage.Contains("解压并校验"))
            || progress.Last().Stage != "更新包已校验，准备应用更新")
            throw new Exception("Missing real transfer and stage progress");
        var chosenDelta = plan.IsDelta;
        var count = plan.Steps.Count;
        var reason = plan.Reason;
        var mode = (string?)input["mode"] ?? "execute";
        bool rolledBack = false;
        if (mode == "directory-rollback")
        {
            var original = Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories).ToDictionary(p => p, DerivedFileUpdate.Hash);
            try
            {
                DerivedFileUpdate.Apply(plan, relative => { if (relative == "zz-final.py") throw new IOException("after directory replacement"); });
                throw new Exception("Expected injected failure");
            }
            catch (IOException) { }
            rolledBack = original.All(p => File.Exists(p.Key) && DerivedFileUpdate.Hash(p.Key) == p.Value)
                && original.Count == Directory.GetFiles(root, "*", SearchOption.AllDirectories).Length
                && Directory.Exists(Path.Combine(root, "reverse/empty"));
            if (!rolledBack) throw new Exception("Directory migration rollback failed");
            await DerivedFileUpdate.Execute(plan, client);
        }
        else if (mode.StartsWith("lock"))
        {
            var original = Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories).ToDictionary(p => p, DerivedFileUpdate.Hash);
            var relative = (string)input["locked"]!;
            using (var held = new FileStream(Path.Combine(root, relative), FileMode.Open, FileAccess.ReadWrite, FileShare.None))
            {
                try { DerivedFileUpdate.Apply(plan); throw new Exception("Expected actual file lock failure"); }
                catch (IOException) { }
            }
            rolledBack = original.All(p => File.Exists(p.Key) && DerivedFileUpdate.Hash(p.Key) == p.Value)
                && original.Count == Directory.GetFiles(root, "*", SearchOption.AllDirectories).Length;
            if (!rolledBack) throw new Exception("Rollback did not restore initial files");
            if (mode == "lock-recover") await DerivedFileUpdate.Execute(plan, client);
        }
        else if (mode == "inject")
        {
            var changes = 0;
            await DerivedFileUpdate.Execute(plan, client, _ => { if (++changes == 5) throw new IOException("fixture halfway failure"); });
        }
        else if (mode != "prepare") await DerivedFileUpdate.Execute(plan, client);
        var output = new { chosen_delta = chosenDelta, count, reason, final_delta = plan.IsDelta, rolled_back = rolledBack,
                           requests = handler.Requests, remaining_budget = plan.Budget.Remaining };
        File.WriteAllText(Path.Combine(work, "result.json"), JsonSerializer.Serialize(output));
        Console.WriteLine(JsonSerializer.Serialize(output));
    }
}
