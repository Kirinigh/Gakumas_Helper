using System;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using MFAAvalonia.Helper;

namespace MFAAvalonia.Helper { } // Allows the same fixture to run against the pre-fix patch.

enum VersionType { Stable, Beta, Alpha }
static class Channels { public static VersionType ToVersionType(this int value) => (VersionType)value; }
sealed class Settings
{
    public int UIUpdateChannelIndex, ResourceUpdateChannelIndex;
    public string GitHubToken = "";
}
static class Instances { public static Settings VersionUpdateSettingsUserControlModel = new(); }
static class LoggerHelper
{
    public static void Info(string message) { }
    public static void Error(string message, Exception? error = null) { }
}
sealed class Handler : HttpMessageHandler
{
    public Func<HttpRequestMessage, HttpResponseMessage> Respond = _ => throw new Exception("Unexpected request");
    public List<string> Requests = new();
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken token)
    {
        Requests.Add(request.RequestUri!.PathAndQuery);
        return Task.FromResult(Respond(request));
    }
}
partial class VersionChecker
{
    internal static Handler Handler = new();
    internal static List<string> Saved = new();
    private static HttpClient CreateHttpClientWithProxy() => new(Handler, false);
    private static void SaveRelease(JToken release, string field) => Saved.Add("release:" + release["tag_name"]);
    private static void SaveChangelog(JToken release, string field) => Saved.Add("changelog:" + release["tag_name"]);
    internal static Task<(string downloadUrl, string sha256)> Detail() => GetDownloadUrlFromGitHubReleaseAsync("v1.0.0", "owner", "repo");
}
static class Program
{
    static void Check(bool condition, string label)
    {
        if (!condition) throw new Exception("FAIL " + label);
        Console.WriteLine("PASS " + label);
    }
    static HttpResponseMessage Json(JToken content) => new(HttpStatusCode.OK) { Content = new StringContent(content.ToString()) };
    static JObject Release(string version, bool pre = false) => new()
    {
        ["tag_name"] = version, ["prerelease"] = pre, ["body"] = "notes",
        ["assets"] = new JArray(new JObject { ["name"] = "resource-win-x64.zip", ["url"] = "https://api.github.com/repos/o/r/releases/assets/1",
            ["browser_download_url"] = "https://github.com/fallback.zip", ["digest"] = "sha256:abc123" })
    };
    static Handler Reset(string token = "")
    {
        Instances.VersionUpdateSettingsUserControlModel = new Settings { GitHubToken = token };
        VersionChecker.Saved.Clear();
        return VersionChecker.Handler = new Handler();
    }
    static async Task<string> Error(Func<Task> action)
    {
        try { await action(); } catch(Exception e) { return e.Message; }
        throw new Exception("Expected failure");
    }
    static async Task Main()
    {
        var h = Reset();
        h.Respond = r => {
            Check(r.Headers.Authorization == null, "anonymous auth unchanged");
            if (r.RequestUri!.AbsolutePath.Contains("/tags/")) return Json(Release("v1.0.65"));
            var query = r.RequestUri.Query.TrimStart('?').Split('&').Select(x => x.Split('=')).ToDictionary(x => x[0], x => int.Parse(x[1]));
            return Json(new JArray(Enumerable.Range(1,65).Skip((query["page"]-1)*query["per_page"]).Take(query["per_page"]).Select(i => Release($"v1.0.{i}"))));
        };
        var found = await VersionChecker.GetLatestVersionAndDownloadUrlFromGithubAsync("o", "r", true);
        Check(found.latestVersion == "v1.0.65" && found.sha256 == "abc123" && found.url.Contains("/assets/1"), "highest version, API asset URL and digest preserved");
        Check(h.Requests.Count == 1 && h.Requests[0].Contains("per_page=100"), $"65 releases require one request, actual={h.Requests.Count}");
        Check(VersionChecker.Saved.SequenceEqual(new[]{"release:v1.0.65"}), "check announcement preserved");

        h = Reset("test-auth");
        h.Respond = r => {
            Check(r.Headers.Authorization?.Parameter == "test-auth", "configured authorization retained");
            return Json(r.RequestUri!.Query.Contains("page=2")
                ? new JArray(Release("v9.0.0"), Release("v12.0.0-beta",true), Release("v13.0.0-alpha",true))
                : new JArray(Enumerable.Range(1,100).Select(i => Release($"v1.0.{i}"))));
        };
        found = await VersionChecker.GetLatestVersionAndDownloadUrlFromGithubAsync("o", "r");
        Check(found.latestVersion == "v9.0.0" && h.Requests.Count == 2, "highest stable can be on later page");
        Check(VersionChecker.Saved.Single() == "changelog:v9.0.0", "download changelog preserved");
        Instances.VersionUpdateSettingsUserControlModel.ResourceUpdateChannelIndex = 1;
        found = await VersionChecker.GetLatestVersionAndDownloadUrlFromGithubAsync("o", "r");
        Check(found.latestVersion == "v12.0.0-beta", "beta excludes alpha");
        Instances.VersionUpdateSettingsUserControlModel.ResourceUpdateChannelIndex = 2;
        found = await VersionChecker.GetLatestVersionAndDownloadUrlFromGithubAsync("o", "r");
        Check(found.latestVersion == "v13.0.0-alpha", "alpha channel retained");
        h.Requests.Clear();
        found = await VersionChecker.GetLatestVersionAndDownloadUrlFromGithubAsync("o", "r", targetVersion:"v1.0.3");
        Check(found.latestVersion == "v1.0.3" && h.Requests.Count == 1, "explicit target reuses matching release assets");

        h = Reset(); h.Respond = _ => Json(new JArray());
        found = await VersionChecker.GetLatestVersionAndDownloadUrlFromGithubAsync("o", "r");
        Check(found.latestVersion == "" && h.Requests.Count == 1, "empty repository has no detail request");
        h = Reset(); h.Respond = r => Json(r.RequestUri!.Query.Contains("page=2") ? new JArray() : new JArray(Enumerable.Range(1,100).Select(i => Release($"v1.0.{i}"))));
        await VersionChecker.GetLatestVersionAndDownloadUrlFromGithubAsync("o", "r");
        Check(h.Requests.Count == 2, "exact full page checks next page");
        h = Reset(); h.Respond = _ => Json(Release("v1.0.0"));
        var asset = await VersionChecker.Detail();
        Check(asset.sha256 == "abc123" && h.Requests.Single().Contains("/releases/tags/"), "standalone detail fetch retained");

        var resume = DateTimeOffset.UtcNow.AddMinutes(20);
        h = Reset("limited");
        h.Respond = _ => {
            var response = new HttpResponseMessage(HttpStatusCode.Forbidden) { ReasonPhrase = "Forbidden", Content = new StringContent("{}") };
            response.Headers.Add("X-RateLimit-Remaining", "0");
            response.Headers.Add("X-RateLimit-Reset", resume.ToUnixTimeSeconds().ToString());
            return response;
        };
        var message = await Error(async () => { await VersionChecker.GetLatestVersionAndDownloadUrlFromGithubAsync("o", "r"); });
        Check(message.Contains(resume.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss")), "primary 403 reports server reset time");
        await Error(async () => { await VersionChecker.GetLatestVersionAndDownloadUrlFromGithubAsync("other", "r"); });
        await Error(async () => { await VersionChecker.Detail(); });
        Check(h.Requests.Count == 1, "cooldown covers subsequent checks and detail across repositories");
        Instances.VersionUpdateSettingsUserControlModel.GitHubToken = "changed-token";
        h.Respond = _ => Json(new JArray());
        await VersionChecker.GetLatestVersionAndDownloadUrlFromGithubAsync("o", "r");
        Check(h.Requests.Count == 2, "changing credentials permits retry");

        foreach (var status in new[]{403,429})
        {
            h = Reset("secondary" + status);
            h.Respond = _ => {
                var response = new HttpResponseMessage((HttpStatusCode)status) { Content = new StringContent("secondary rate limit exceeded") };
                response.Headers.RetryAfter = new RetryConditionHeaderValue(TimeSpan.FromSeconds(1));
                return response;
            };
            await Error(async () => { await VersionChecker.Detail(); });
            await Error(async () => { await VersionChecker.Detail(); });
            Check(h.Requests.Count == 1, $"{status} Retry-After suppresses immediate retries");
            await Task.Delay(1150);
            h.Respond = _ => Json(Release("v1.0.0"));
            await VersionChecker.Detail();
            Check(h.Requests.Count == 2, $"{status} resumes after cooldown");
        }
        h = Reset("denied");
        h.Respond = _ => new HttpResponseMessage(HttpStatusCode.Forbidden) { Content = new StringContent("Resource not accessible by integration") };
        message = await Error(async () => { await VersionChecker.Detail(); });
        await Error(async () => { await VersionChecker.Detail(); });
        Check(!message.Contains("额度受限") && h.Requests.Count == 2, "ordinary permission denial is not rate limiting");
        h = Reset("body-only");
        h.Respond = _ => new HttpResponseMessage(HttpStatusCode.Forbidden) { Content = new StringContent("API rate limit exceeded") };
        message = await Error(async () => { await VersionChecker.Detail(); });
        await Error(async () => { await VersionChecker.Detail(); });
        Check(message.Contains("额度受限") && h.Requests.Count == 1, "body-only rate limit uses default cooldown");
        Console.WriteLine("All GitHub update regression cases passed; no network used.");
    }
}
