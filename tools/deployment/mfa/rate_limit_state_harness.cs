using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MFAAvalonia.Helper;

namespace MFAAvalonia.Helper
{
    static class LoggerHelper
    {
        internal static readonly List<string> Messages = new();
        public static void Info(string message) => Messages.Add(message);
    }
}

sealed class ResponseHandler : HttpMessageHandler
{
    internal int Calls;
    internal Func<HttpResponseMessage> Response = () => new(HttpStatusCode.OK);
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken token)
    {
        Calls++;
        return Task.FromResult(Response());
    }
}

static class Program
{
    static async Task<bool> Request(HttpClient client, string url)
    {
        try
        {
            using var response = await GitHubApiRequests.GetAsync(client, url);
            if (!response.IsSuccessStatusCode) throw new Exception("Unexpected response");
            return false;
        }
        catch (Exception error) when (error.Message.StartsWith("GitHub 请求额度受限")) { return true; }
    }

    static async Task Main(string[] args)
    {
        var mode = args[0];
        var delay = double.Parse(args[1], System.Globalization.CultureInfo.InvariantCulture);
        var root = Path.Combine(AppContext.BaseDirectory, ".local", "runtime-data");
        var state = Path.Combine(root, "github-api-rate-limit.json");
        if (mode == "hold-lock")
        {
            Directory.CreateDirectory(root);
            using var fileLock = new FileStream(Path.Combine(root, "github-api-rate-limit.lock"),
                FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite);
            fileLock.Lock(0, 1);
            try
            {
                File.WriteAllText(Path.Combine(AppContext.BaseDirectory, "locked.ready"), "ready");
                var until = Stopwatch.StartNew();
                while (!File.Exists(Path.Combine(AppContext.BaseDirectory, "go")))
                {
                    if (until.Elapsed.TotalSeconds > 10) throw new Exception("Lock barrier timed out");
                    Thread.Sleep(10);
                }
            }
            finally { fileLock.Unlock(0, 1); }
            Console.WriteLine("{}");
            return;
        }
        var handler = new ResponseHandler();
        using var client = new HttpClient(handler);
        if (args[2] == "auth") client.DefaultRequestHeaders.Authorization = new("token", "fixture-auth");
        var url = args[3] == "github" ? "https://api.github.com/test" : "https://example.org/test";
        handler.Response = () =>
        {
            if (mode == "race")
            {
                Directory.CreateDirectory(root);
                File.WriteAllText(state, JsonSerializer.Serialize(new
                {
                    schema_version = 1, resume_at = DateTimeOffset.UtcNow.ToUnixTimeSeconds() + 3600
                }));
            }
            if (mode == "barrier")
            {
                File.WriteAllText(Path.Combine(AppContext.BaseDirectory, args[4]), "ready");
                var until = Stopwatch.StartNew();
                while (!File.Exists(Path.Combine(AppContext.BaseDirectory, "go")))
                {
                    if (until.Elapsed.TotalSeconds > 10) throw new Exception("Barrier timed out");
                    Thread.Sleep(10);
                }
            }
            if (mode == "success" || mode == "corrupt-after") return new HttpResponseMessage(HttpStatusCode.OK);
            var limited = new HttpResponseMessage((HttpStatusCode)429) { Content = new StringContent("limited") };
            limited.Headers.RetryAfter = new RetryConditionHeaderValue(TimeSpan.FromSeconds(delay));
            return limited;
        };
        var timer = Stopwatch.StartNew();
        var first = await Request(client, url);
        if (mode == "corrupt-after") File.WriteAllText(state, "broken");
        var second = await Request(client, url);
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            calls = handler.Calls, first_limited = first, second_limited = second,
            diagnostics = LoggerHelper.Messages.Count, elapsed_ms = timer.Elapsed.TotalMilliseconds
        }));
    }
}
