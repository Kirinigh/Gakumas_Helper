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
    }
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
static class Program
{
    static async Task Main(string[] args)
    {
        if (args.Length == 2 && args[0] == "--launch-plan")
        {
            var prepared = JsonSerializer.Deserialize<DerivedFileUpdate.Plan>(File.ReadAllText(args[1]))!;
            await DerivedFileUpdate.LaunchWorker(prepared, "", "");
            return;
        }
        var input = JsonNode.Parse(File.ReadAllText(args[0]))!;
        var root = (string)input["root"]!;
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
        if (mode.StartsWith("lock"))
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
