using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
using System.Threading.Tasks;
using MFAAvalonia.Helper;

// The production FileLogExporter.cs is compiled as a separate, unchanged input.
internal static class Program
{
    private static int _passed;
    private static string _root = "";
    private const string Agent = ".local/runtime-data/logs/";
    private const string Failure = ".local/arena-win-rate/reader-failures/txn/";
    private static readonly byte[] Png = Convert.FromBase64String(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jP1kAAAAASUVORK5CYII=");

    private static void Assert(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }

    private static async Task Case(string name, Func<string, Task> test)
    {
        var input = Path.Combine(_root, name, "input");
        Directory.CreateDirectory(input);
        AppPaths.DataRoot = input;
        ToastHelper.Messages.Clear();
        LoggerHelper.Messages.Clear();
        await test(input);
        _passed++;
        Console.WriteLine("PASS " + name);
    }

    private static void Write(string root, string relative, byte[] value, int ageDays = 0)
    {
        var path = Path.Combine(root, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, value);
        if (ageDays != 0) File.SetLastWriteTime(path, DateTime.Now.AddDays(-ageDays));
    }

    private static void Write(string root, string relative, string value, int ageDays = 0)
        => Write(root, relative, Encoding.UTF8.GetBytes(value), ageDays);

    private static byte[] LogZip(string name, string text)
    {
        using var memory = new MemoryStream();
        using (var archive = new ZipArchive(memory, ZipArchiveMode.Create, true))
        {
            using var writer = new StreamWriter(archive.CreateEntry(name).Open(), new UTF8Encoding(false));
            writer.Write(text);
        }
        return memory.ToArray();
    }

    private static void Populate(string root)
    {
        Write(root, Agent + "2026-09-10.log", "ERROR 位置=己方/舞台1/成员2/组2\n{\"event\":\"arena_member_read_attempt\",\"succeeded\":false}\n");
        Write(root, Agent + "current.log", "fallback current log\n");
        Write(root, Agent + "2026-09-09.log.zip", LogZip("2026-09-09.log", "loguru rotation"));
        Write(root, Agent + "current.log.2026-09-09.zip", LogZip("current.log.2026-09-09", "stdlib rotation"));
        Write(root, Failure + "evidence.json", "{\"event\":\"arena_detail_failure\",\"stage_number\":1,\"member_slot\":2}");
        Write(root, Failure + "00.png", Png);
        Write(root, "debug/maafw.log", "native log");
        Write(root, "debug/on_error/failed.png", Png);
        Write(root, "debug/vision/ocr.png", Png);
        Write(root, "debug/other.png", Png);
        Write(root, "logs/log-20260910.log", "concise GUI log");
        Write(root, "debug/custom.log", "old custom log");
        Write(root, ".local/arena-win-rate/own-score-cache.json", "must remain private");
        Write(root, ".local/arena-win-rate/results/battle.json", "not a diagnostic");
        Write(root, ".local/runtime-data/unrelated.json", "not a log");
        Write(root, Agent + "unrelated.zip", "not a rotated log");
        Write(root, Failure + "unrelated.bin", "not failure evidence");
    }

    private static ExportLogPackageOptions OnlyCustom() => new()
    {
        IncludeMaaLog = false, IncludeGuiLog = false, IncludeCustomLog = true,
        IncludeOnErrorImages = false, IncludeVisionImages = false, IncludeOtherImages = false
    };

    private static async Task WithUnreadableDirectory(string path, Func<Task> test)
    {
        var directory = Directory.CreateDirectory(path);
        var original = directory.GetAccessControl(AccessControlSections.Access);
        var denied = directory.GetAccessControl(AccessControlSections.Access);
        var user = WindowsIdentity.GetCurrent().User ?? throw new InvalidOperationException("Missing test user SID");
        denied.AddAccessRule(new FileSystemAccessRule(user,
            FileSystemRights.ListDirectory | FileSystemRights.ReadAttributes, AccessControlType.Deny));
        try
        {
            directory.SetAccessControl(denied);
            await test();
        }
        finally { directory.SetAccessControl(original); }
    }

    private static async Task<(ExportLogResult Result, Dictionary<string, byte[]> Entries)> Export(
        string input, ExportLogPackageOptions? options)
    {
        var output = Path.Combine(Path.GetDirectoryName(input)!, "output.zip");
        var result = await FileLogExporter.CompressRecentLogs(new Picker(output), options);
        var entries = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        if (File.Exists(output))
        {
            using var archive = ZipFile.OpenRead(output);
            foreach (var entry in archive.Entries)
            {
                using var reader = entry.Open();
                using var memory = new MemoryStream();
                reader.CopyTo(memory);
                entries.Add(entry.FullName, memory.ToArray());
            }
        }
        return (result, entries);
    }

    private static void Exact(Dictionary<string, byte[]> entries, string input, params string[] names)
    {
        Assert(entries.Keys.Order().SequenceEqual(names.Append("export-report.txt").Order()),
            "Unexpected ZIP members: " + string.Join(", ", entries.Keys));
        foreach (var name in names)
        {
            using var source = new FileStream(Path.Combine(input, name), FileMode.Open, FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete);
            using var memory = new MemoryStream();
            source.CopyTo(memory);
            Assert(entries[name].SequenceEqual(memory.ToArray()), "Changed bytes: " + name);
        }
        var report = Encoding.UTF8.GetString(entries["export-report.txt"]);
        Assert(!report.Contains(input), "Report must use relative paths");
        Assert(report.Contains("已包含文件"), "Missing readable export report");
    }

    private static async Task Main(string[] args)
    {
        try { await Run(args); }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            Environment.ExitCode = 1;
        }
    }

    private static async Task Run(string[] args)
    {
        _root = args.Single();
        Directory.CreateDirectory(_root);
        await Case("default_real_zip", async input =>
        {
            Populate(input);
            var (result, entries) = await Export(input, new());
            Assert(result == ExportLogResult.Success, "Default export failed");
            Exact(entries, input, Agent + "2026-09-10.log", Agent + "current.log",
                Agent + "2026-09-09.log.zip", Agent + "current.log.2026-09-09.zip",
                Failure + "evidence.json", Failure + "00.png", "debug/maafw.log",
                "debug/custom.log", "debug/on_error/failed.png", "debug/other.png", "logs/log-20260910.log");
        });
        await Case("legacy_real_zip_five_day_limit", async input =>
        {
            Populate(input);
            Write(input, Agent + "old.log", "old", 6);
            Write(input, Failure + "old.png", Png, 6);
            var (result, entries) = await Export(input, null);
            Assert(result == ExportLogResult.Success, "Legacy export failed");
            Exact(entries, input, Agent + "2026-09-10.log", Agent + "current.log",
                Agent + "2026-09-09.log.zip", Agent + "current.log.2026-09-09.zip",
                Failure + "evidence.json", Failure + "00.png", "debug/maafw.log",
                "debug/custom.log", "debug/on_error/failed.png", "debug/other.png", "logs/log-20260910.log");
        });
        await Case("custom_only_both_backends_and_zip_bytes", async input =>
        {
            Populate(input);
            var (result, entries) = await Export(input, OnlyCustom());
            Assert(result == ExportLogResult.Success, "Custom-only export failed");
            Exact(entries, input, Agent + "2026-09-10.log", Agent + "current.log",
                Agent + "2026-09-09.log.zip", Agent + "current.log.2026-09-09.zip",
                Failure + "evidence.json", "debug/custom.log");
        });
        await Case("custom_unchecked", async input =>
        {
            Populate(input);
            var (result, entries) = await Export(input, new() { IncludeCustomLog = false });
            Assert(result == ExportLogResult.Success, "Export without custom failed");
            Exact(entries, input, Failure + "00.png", "debug/maafw.log", "debug/on_error/failed.png",
                "debug/other.png", "logs/log-20260910.log");
        });
        await Case("failure_images_unchecked_metadata_kept", async input =>
        {
            Populate(input);
            var (result, entries) = await Export(input, new() { IncludeOnErrorImages = false });
            Assert(result == ExportLogResult.Success, "Export without failure images failed");
            Assert(entries.ContainsKey(Failure + "evidence.json"), "Failure metadata is a custom log");
            Assert(!entries.ContainsKey(Failure + "00.png") && !entries.ContainsKey("debug/on_error/failed.png"),
                "Failure pictures ignored the checkbox");
            Assert(entries.ContainsKey("debug/other.png"), "Other pictures lost");
        });
        foreach (var (range, maximumAge) in new[] {
                     (ExportLogTimeRange.Last24Hours, 0), (ExportLogTimeRange.Last3Days, 2),
                     (ExportLogTimeRange.Last7Days, 4), (ExportLogTimeRange.All, 8) })
        {
            await Case("failure_time_" + range, async input =>
            {
                foreach (var age in new[] { 0, 2, 4, 8 })
                {
                    Write(input, Failure + age + ".png", Png, age);
                    Write(input, "debug/on_error/" + age + ".png", Png, age);
                }
                var (result, entries) = await Export(input, new()
                {
                    IncludeMaaLog = false, IncludeGuiLog = false, IncludeCustomLog = false,
                    IncludeOnErrorImages = true, IncludeVisionImages = false, IncludeOtherImages = false,
                    OnErrorImageTimeRange = range
                });
                Assert(result == ExportLogResult.Success, "Time-filter export failed");
                foreach (var age in new[] { 0, 2, 4, 8 })
                    foreach (var prefix in new[] { Failure, "debug/on_error/" })
                        Assert(entries.ContainsKey(prefix + age + ".png") == (age <= maximumAge), "Wrong failure image time range");
            });
        }
        await Case("vision_and_other_time_options_unchanged", async input =>
        {
            Write(input, "debug/vision/now.png", Png);
            Write(input, "debug/vision/old.png", Png, 2);
            Write(input, "debug/now.png", Png);
            Write(input, "debug/old.png", Png, 2);
            Write(input, Failure + "00.png", Png);
            var (result, entries) = await Export(input, new()
            {
                IncludeMaaLog = false, IncludeGuiLog = false, IncludeCustomLog = false,
                IncludeOnErrorImages = false, IncludeVisionImages = true, IncludeOtherImages = true,
                VisionImageTimeRange = ExportLogTimeRange.Last24Hours,
                OtherImageTimeRange = ExportLogTimeRange.Last24Hours
            });
            Assert(result == ExportLogResult.Success, "Image-only export failed");
            Exact(entries, input, "debug/vision/now.png", "debug/now.png");
        });
        await Case("shared_active_log", async input =>
        {
            var path = Agent + "2026-09-10.log";
            Write(input, path, "first\n");
            using var active = new FileStream(Path.Combine(input, path), FileMode.Append, FileAccess.Write,
                FileShare.ReadWrite | FileShare.Delete);
            var bytes = Encoding.UTF8.GetBytes("second 详细错误\n");
            active.Write(bytes);
            active.Flush(true);
            var (result, entries) = await Export(input, OnlyCustom());
            Assert(result == ExportLogResult.Success, "Shared active log was skipped");
            Exact(entries, input, path);
        });
        await Case("locked_file_partial_warning_and_report", async input =>
        {
            Populate(input);
            var blocked = Agent + "2026-09-10.log";
            using var exclusive = new FileStream(Path.Combine(input, blocked), FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            var (result, entries) = await Export(input, OnlyCustom());
            Assert(result == ExportLogResult.Partial, "A missing required log was reported as complete");
            Assert(!entries.ContainsKey(blocked), "Locked log must not be empty or partially copied");
            Assert(entries.ContainsKey(Agent + "current.log"), "Other diagnostics were discarded");
            Assert(ToastHelper.Messages.Any(x => x.StartsWith("WARN:") && x.Contains("部分导出")), "Missing partial warning");
            Assert(!ToastHelper.Messages.Any(x => x.StartsWith("SUCCESS:")), "Misleading success notification");
            var report = Encoding.UTF8.GetString(entries["export-report.txt"]).Replace('\\', '/');
            Assert(report.Contains(blocked) && report.Contains("IOException") && report.Contains("部分导出"),
                "The ZIP did not explain the missing file");
        });
        await Case("all_selected_files_locked_fails", async input =>
        {
            var path = Agent + "current.log";
            Write(input, path, "cannot export");
            using var exclusive = new FileStream(Path.Combine(input, path), FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            var (result, entries) = await Export(input, OnlyCustom());
            Assert(result == ExportLogResult.Failed && entries.Count == 0, "No payload is not a successful export");
        });
        await Case("selected_agent_root_unreadable_is_partial", async input =>
        {
            Populate(input);
            await WithUnreadableDirectory(Path.Combine(input, Agent), async () =>
            {
                var (result, entries) = await Export(input, new());
                Assert(result == ExportLogResult.Partial, "Unreadable Agent log root was treated as absent");
                var report = Encoding.UTF8.GetString(entries["export-report.txt"]).Replace('\\', '/');
                Assert(report.Contains(".local/runtime-data/logs: UnauthorizedAccessException"), "Missing unreadable root explanation");
            });
        });
        await Case("unrelated_unreadable_cache_does_not_make_partial", async input =>
        {
            Populate(input);
            await WithUnreadableDirectory(Path.Combine(input, ".local/unrelated-cache"), async () =>
            {
                var (result, entries) = await Export(input, new());
                Assert(result == ExportLogResult.Success, "Unselected unreadable cache was treated as a missing log");
                Assert(entries.ContainsKey(Agent + "2026-09-10.log"), "Known log was lost");
            });
        });
        await Case("only_selected_root_unreadable_explains_failure", async input =>
        {
            Write(input, Agent + "current.log", "hidden failure");
            await WithUnreadableDirectory(Path.Combine(input, Agent), async () =>
            {
                var (result, entries) = await Export(input, OnlyCustom());
                Assert(result == ExportLogResult.Failed && entries.Count == 0, "Unreadable selection became empty success");
                Assert(ToastHelper.Messages.Any(x => x.StartsWith("ERROR:") && x.Contains("日志读取失败")
                    && x.Contains("runtime-data")), "Missing read failure notification");
                Assert(!ToastHelper.Messages.Any(x => x.Contains("ExportLogNoEligibleFiles")), "Unreadable logs shown as absent");
            });
        });
        await Case("unselected_reader_failure_root_not_read", async input =>
        {
            Populate(input);
            await WithUnreadableDirectory(Path.Combine(input, ".local/arena-win-rate/reader-failures"), async () =>
            {
                var (result, entries) = await Export(input, new()
                {
                    IncludeCustomLog = false, IncludeOnErrorImages = false
                });
                Assert(result == ExportLogResult.Success, "Unselected failure root was read");
                Assert(!entries.Keys.Any(x => x.StartsWith(".local/arena-win-rate/")), "Unselected failure data was included");
            });
        });
        await Case("default_unselected_vision_not_read", async input =>
        {
            Populate(input);
            await WithUnreadableDirectory(Path.Combine(input, "debug/vision"), async () =>
            {
                var (result, entries) = await Export(input, new());
                Assert(result == ExportLogResult.Success, "Unselected vision directory caused partial export");
                Assert(entries.ContainsKey("debug/maafw.log"), "Framework log lost while skipping vision");
            });
        });
        await Case("on_error_ancestor_takes_priority_over_vision", async input =>
        {
            Write(input, "debug/on_error/vision/failed.png", Png);
            var (result, entries) = await Export(input, new());
            Assert(result == ExportLogResult.Success, "Nested on_error export failed");
            Exact(entries, input, "debug/on_error/vision/failed.png");
        });
        await Case("failure_image_without_metadata_explained", async input =>
        {
            Write(input, Failure + "00.png", Png);
            var (result, entries) = await Export(input, new());
            Assert(result == ExportLogResult.Partial, "Unwritten selected failure metadata was silently ignored");
            var report = Encoding.UTF8.GetString(entries["export-report.txt"]);
            Assert(report.Contains("evidence.json") && report.Contains("FileNotFoundException"), "Missing metadata is unexplained");
        });
        await Case("large_text_not_truncated", async input =>
        {
            var path = Agent + "current.log";
            Write(input, path, string.Concat(Enumerable.Repeat("detail record\n", FileLogExporter.MAX_LINES + 50)) + "LAST ERROR 卡位\n");
            var (result, entries) = await Export(input, OnlyCustom());
            Assert(result == ExportLogResult.Success, "Large log failed");
            Exact(entries, input, path);
        });
        await Case("legacy_native_selection_preserved", async input =>
        {
            Write(input, "debug/maa.log", "named");
            Write(input, "debug/extra.log", "legacy debug");
            var (result, entries) = await Export(input, null);
            Assert(result == ExportLogResult.Success, "Legacy native export failed");
            Exact(entries, input, "debug/maa.log", "debug/extra.log");
        });
        await Case("normal_named_native_selection_preserved", async input =>
        {
            Write(input, "debug/maa.log", "named");
            Write(input, "debug/extra.log", "legacy debug");
            var (result, entries) = await Export(input, new() { IncludeCustomLog = false });
            Assert(result == ExportLogResult.Success, "Normal native export failed");
            Exact(entries, input, "debug/maa.log");
        });
        await Case("absent_failure_directory_is_normal", async input =>
        {
            Write(input, Agent + "current.log", "no member failure");
            var (result, entries) = await Export(input, new());
            Assert(result == ExportLogResult.Success, "Missing optional evidence became a false failure");
            var report = Encoding.UTF8.GetString(entries["export-report.txt"]);
            Assert(report.Contains("失败证据目录存在：False"), "Absent evidence directory was not explained");
        });
        await Case("no_eligible_files", async input =>
        {
            var (result, entries) = await Export(input, new());
            Assert(result == ExportLogResult.NoEligibleFiles && entries.Count == 0, "Empty selection handling changed");
        });
        Console.WriteLine($"PASS {_passed} real-export cases; ZIP bytes and report contents verified.");
    }

    private sealed class Picker(string output) : Avalonia.Platform.Storage.IStorageProvider
    {
        public Task<Avalonia.Platform.Storage.IStorageFile?> SaveFilePickerAsync(
            Avalonia.Platform.Storage.FilePickerSaveOptions options)
            => Task.FromResult<Avalonia.Platform.Storage.IStorageFile?>(new StorageFile(output));
    }

    private sealed class StorageFile(string path) : Avalonia.Platform.Storage.IStorageFile
    {
        public string Name => System.IO.Path.GetFileName(path);
        public Uri Path => new(System.IO.Path.GetFullPath(path));
        public Task<Stream> OpenWriteAsync()
            => Task.FromResult<Stream>(new FileStream(path, FileMode.Create, FileAccess.Write));
    }
}

namespace Avalonia.Platform.Storage
{
    public interface IStorageProvider { Task<IStorageFile?> SaveFilePickerAsync(FilePickerSaveOptions options); }
    public interface IStorageFile { string Name { get; } Uri Path { get; } Task<Stream> OpenWriteAsync(); }
    public sealed class FilePickerSaveOptions
    {
        public string? Title { get; init; }
        public string? DefaultExtension { get; init; }
        public string? SuggestedFileName { get; init; }
    }
}
namespace MFAAvalonia.Extensions
{
    public static class TextExtensions { public static string ToLocalization(this string value) => value; }
}
namespace MFAAvalonia.Extensions.MaaFW { public sealed class UnusedNamespaceAnchor { } }
namespace MFAAvalonia.Helper
{
    public static class AppPaths { public static string DataRoot { get; set; } = ""; }
    public static class Instances { public static RootModel RootViewModel { get; } = new(); }
    public sealed class RootModel { public bool IsRunning { get; set; } }
    public static class LoggerHelper
    {
        public static List<string> Messages { get; } = new();
        public static void Info(string text) => Messages.Add(text);
        public static void Warning(string text) => Messages.Add(text);
        public static void Error(string text, Exception? error = null) => Messages.Add(text);
    }
    public static class ToastHelper
    {
        public static List<string> Messages { get; } = new();
        public static void Info(string title, string body) => Messages.Add("INFO:" + body);
        public static void Warn(string title, string body) => Messages.Add("WARN:" + body);
        public static void Error(string title, string body) => Messages.Add("ERROR:" + body);
        public static void Success(string title, string body) => Messages.Add("SUCCESS:" + body);
    }
    public static class LangKeys
    {
        public const string Warning = "Warning", StopTaskBeforeExportLog = "StopTaskBeforeExportLog",
            ExportLog = "ExportLog", ExportLogFailed = "ExportLogFailed",
            ExportLogNoEligibleFiles = "ExportLogNoEligibleFiles", ExportLogSuccess = "ExportLogSuccess",
            ExportLogInProgress = "ExportLogInProgress";
    }
}
