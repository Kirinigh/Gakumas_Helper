using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json;

public class AvaloniaList<T> : List<T> { }
public static class AppPaths { public static string DataRoot = ""; }
public static class ConfigurationKeys
{
    public const string EnableLiveView = "UI.LiveView.EnableLiveView";
    public const string DashboardCardGridLayout = "layout";
    public static readonly string[] InstanceScopedKeys = [EnableLiveView, "CurrentController", "BeforeTask"];
}
public static class LoggerHelper { public static void Info(object value) { } public static void Warning(object value) { } }
public static class JsonHelper
{
    public static T LoadJson<T>(string path, T fallback) => File.Exists(path) ? JsonConvert.DeserializeObject<T>(File.ReadAllText(path))! : fallback;
    public static void SaveJson(string path, object value, params object[] converters) => File.WriteAllText(path, JsonConvert.SerializeObject(value));
    public static void SaveConfig(string name, object value, params object[] converters) => SaveJson(Path.Combine(AppPaths.DataRoot, "config", name + ".json"), value);
    public static T LoadConfig<T>(string name, T fallback)
    {
        var path = Path.Combine(AppPaths.DataRoot, "config", name + ".json");
        return File.Exists(path) ? JsonConvert.DeserializeObject<T>(File.ReadAllText(path))! : fallback;
    }
}
public class MaaInterfaceSelectAdvancedConverter(bool enabled) { }
public class MaaInterfaceSelectOptionConverter(bool enabled) { }
public partial class MaaProcessorManager
{
    public void Migrate(string directory) => MigratePlainKeysToDefaultInstance(directory);
}
public class MFAConfiguration(string name, string fileName, Dictionary<string, object> config)
{
    public string Name = name;
    public string FileName = fileName;
    public Dictionary<string, object> Config = config;
    public int Writes;
    public MFAConfiguration SetConfig(Dictionary<string, object> value) { Config = value; return this; }
    public bool ContainsKey(string key) => Config.ContainsKey(key);
    public T GetValue<T>(string key, T fallback) => Config.TryGetValue(key, out var value)
        ? JsonConvert.DeserializeObject<T>(JsonConvert.SerializeObject(value))! : fallback;
    public void SetValue(string key, object value) { Config[key] = value; Writes++; }
}
public partial class ConfigurationManager
{
    private static string ConfigDir => Path.Combine(AppPaths.DataRoot, "config");
    public static string ConfigName = "Default";
    public static string Selected = "Default";
    public static MFAConfiguration Current = new("Default", "config", new());
    public static MFAConfiguration Maa = new("Maa", "maa_option", new());
    public static AvaloniaList<MFAConfiguration> Configs = new();
    private static string GetDefaultConfig() => Selected;
    public static void Reload()
    {
        Current = new("Default", "config", new());
        Configs = LoadConfigurations();
    }
}
public partial class InstanceConfiguration
{
    private string _instanceId;
    private Dictionary<string, object> _config;
    private MFAConfiguration GlobalConfig => ConfigurationManager.Current;
    public InstanceConfiguration(string id, Dictionary<string, object> values) { _instanceId = id; _config = values; }
    private T? ConvertValue<T>(object data) => JsonConvert.DeserializeObject<T>(JsonConvert.SerializeObject(data));
    private void PersistFallbackValue(string key, object? value) { if (value != null) _config[key] = value; }
}
public class DashboardCardLayout { public string Id = ""; public int Row; public int Col; }
public class DashboardGridMeta { public int Rows; public int Columns; }
public class ResourceLayoutDefinition { public int Rows = 8; public int Columns = 12; }
public partial class DashboardCardGrid
{
    private bool _layoutLoaded;
    public string Key = "layout.grid";
    public string ResourceHash = "new-default";
    public ResourceLayoutDefinition? Resource = new();
    public DashboardGridMeta? Meta;
    public List<DashboardCardLayout>? Applied;
    public DashboardGridMeta? AppliedMeta;
    public int ResourcesApplied, Saves, ResourceReads, FilesGenerated;
    private string GetLayoutKey() => Key;
    private string GetResourceLayoutHashKey() => "resource-hash";
    private List<DashboardCardLayout> GetDefaultLayouts() => new() { new() { Id = "builtin" } };
    private DashboardGridMeta? LoadLayoutMeta() => Meta;
    private ResourceLayoutDefinition? TryLoadResourceLayout(out string hash) { ResourceReads++; hash = ResourceHash; return Resource; }
    private void ApplyResourceLayout(ResourceLayoutDefinition value, List<DashboardCardLayout> defaults) { ResourcesApplied++; Applied = new() { new() { Id = "resource" } }; }
    private void SaveLayouts() { Saves++; ConfigurationManager.Current.SetValue(Key, Applied!); }
    private void ApplyLayoutMeta(DashboardGridMeta? value) => AppliedMeta = value;
    private void ApplyLayouts(List<DashboardCardLayout> value) => Applied = value;
    private void EnsureResourceLayoutFile(List<DashboardCardLayout> value) => FilesGenerated++;
    public void Load() => EnsureLayoutsLoaded();
}
public static class Program
{
    private static int Checks;
    private static void Check(bool ok, string message) { Checks++; if (!ok) throw new Exception(message); }
    private static string Key => ConfigurationKeys.EnableLiveView;
    private static void Write(string name, object value)
    {
        File.WriteAllText(Path.Combine(AppPaths.DataRoot, "config", name + ".json"), JsonConvert.SerializeObject(value));
    }
    public static void Main(string[] args)
    {
        AppPaths.DataRoot = args[0];
        Directory.CreateDirectory(Path.Combine(args[0], "config"));
        var template = Path.Combine(args[0], "config.template.json");
        File.WriteAllText(template, JsonConvert.SerializeObject(new Dictionary<string, object> { [Key] = false, ["NoAutoStart"] = true }));
        ConfigurationManager.Reload();
        Check(!ConfigurationManager.Current.GetValue(Key, true), "clean startup uses template");
        Check(!ConfigurationManager.Current.ContainsKey("NoAutoStart"), "unrelated template key ignored");
        Check(File.ReadAllText(Path.Combine(args[0], "config/config.json")) == "{}", "loading does not rewrite user configuration");
        // First-start migration moves plain instance keys to a temporary default
        // instance, then discards that instance before creating interface tabs.
        var instancesDir = Path.Combine(args[0], "config", "instances");
        Directory.CreateDirectory(instancesDir);
        var bootstrapPath = Path.Combine(instancesDir, "default.json");
        new MaaProcessorManager().Migrate(instancesDir);
        if (File.Exists(bootstrapPath)) File.Delete(bootstrapPath);
        Check(!new InstanceConfiguration("from-interface", new()).GetValue(Key, true), "template survives discarded bootstrap instance");
        Check(new InstanceConfiguration("from-interface", new() { [Key] = true }).GetValue(Key, false), "explicit instance choice overrides template after migration");
        Check(!new InstanceConfiguration("from-interface", new()).GetValue("NoAutoStart", false), "instance fallback ignores unrelated template keys");
        foreach (var enabled in new[] { false, true })
        {
            ConfigurationManager.Current.Config[Key] = enabled;
            ConfigurationManager.Current.Config["CurrentController"] = "Win32";
            new MaaProcessorManager().Migrate(instancesDir);
            Check(!ConfigurationManager.Current.ContainsKey("CurrentController"), "other plain settings keep existing migration");
            Check(File.ReadAllText(bootstrapPath).Contains("Win32"), "other migrated values are retained");
            File.Delete(bootstrapPath);
            Check(new InstanceConfiguration("fresh-tab", new()).GetValue(Key, !enabled) == enabled, "explicit global choice survives bootstrap deletion");
            Check(new InstanceConfiguration("fresh-tab", new() { [Key] = !enabled }).GetValue(Key, enabled) == !enabled, "explicit local choice still wins after migration");
        }
        Check(!ConfigurationManager.Add("new-profile").GetValue(Key, true), "new profile uses same default");
        foreach (var enabled in new[] { true, false })
        {
            Write("config", new Dictionary<string, object> { [Key] = enabled, ["NoAutoStart"] = false, ["Tasks"] = "keep" });
            var before = File.ReadAllBytes(Path.Combine(args[0], "config/config.json"));
            ConfigurationManager.Reload();
            Check(ConfigurationManager.Current.GetValue(Key, !enabled) == enabled, "existing value wins");
            Check(!ConfigurationManager.Current.GetValue("NoAutoStart", true), "NoAutoStart retained");
            Check(before.SequenceEqual(File.ReadAllBytes(Path.Combine(args[0], "config/config.json"))), "existing bytes retained");
        }
        Write("config", new Dictionary<string, object> { ["NoAutoStart"] = false, ["Instance.old." + Key] = true });
        Write("mfa_named", new Dictionary<string, object> { [Key] = true });
        ConfigurationManager.Reload();
        Check(!ConfigurationManager.Current.GetValue(Key, true), "missing key gets default");
        Check(ConfigurationManager.Configs.Single(x => x.Name == "named").GetValue(Key, false), "named profile retained");
        Check(new InstanceConfiguration("old", new()).GetValue(Key, false), "legacy scoped user choice wins");
        Check(new InstanceConfiguration("local", new() { [Key] = true }).GetValue(Key, false), "instance user choice wins");
        Check(!new InstanceConfiguration("fresh", new()).GetValue(Key, true), "instance fallback gets template");
        ConfigurationManager.Selected = "named";
        ConfigurationManager.Reload();
        Check(ConfigurationManager.Current.Name == "named" && ConfigurationManager.Current.GetValue(Key, false), "selected named profile retained");
        ConfigurationManager.Selected = "Default";
        foreach (var invalid in new[] { "{", "[]", "{\"UI.LiveView.EnableLiveView\":\"false\"}", "{}" })
        {
            File.WriteAllText(template, invalid);
            Write("config", new Dictionary<string, object> { ["NoAutoStart"] = false });
            ConfigurationManager.Reload();
            Check(!ConfigurationManager.Current.ContainsKey(Key), "invalid optional template leaves default unchanged");
            Check(!ConfigurationManager.Current.GetValue("NoAutoStart", true), "invalid template retains unrelated settings");
        }
        File.Delete(template);
        ConfigurationManager.Reload();
        Check(!ConfigurationManager.Current.ContainsKey(Key), "absent template supported");
        foreach (var legacy in new[] { false, true })
        {
            var persisted = new List<DashboardCardLayout> { new() { Id = "user", Row = 3, Col = 2 } };
            ConfigurationManager.Current = new("Default", "config", new() { [legacy ? "layout" : "layout.grid"] = persisted, ["resource-hash"] = "old" });
            var saved = JsonConvert.SerializeObject(ConfigurationManager.Current.Config);
            var grid = new DashboardCardGrid { Meta = new() { Rows = 5, Columns = 7 } };
            grid.Load(); grid.Load();
            Check(grid.Applied![0].Id == "user" && grid.Applied[0].Row == 3, "saved layout retained across changed resource");
            Check(grid.AppliedMeta == grid.Meta, "saved dimensions retained");
            Check(grid.ResourceReads == 0 && grid.ResourcesApplied == 0 && grid.Saves == 0, "saved layout bypasses resource replacement");
            Check(saved == JsonConvert.SerializeObject(ConfigurationManager.Current.Config), "saved dictionary unchanged");
        }
        ConfigurationManager.Current = new("Default", "config", new());
        var fresh = new DashboardCardGrid(); fresh.Load(); fresh.Load();
        Check(fresh.ResourcesApplied == 1 && fresh.Saves == 1, "clean layout applies resource once");
        Check(ConfigurationManager.Current.ContainsKey("layout.grid"), "resource default saved using existing layout key");
        var restarted = new DashboardCardGrid { ResourceHash = "later" }; restarted.Load();
        Check(restarted.ResourcesApplied == 0 && restarted.Saves == 0, "restart preserves saved layout");
        ConfigurationManager.Current = new("Default", "config", new());
        var noResource = new DashboardCardGrid { Resource = null }; noResource.Load();
        Check(noResource.Applied![0].Id == "builtin" && noResource.FilesGenerated == 1, "no resource keeps builtin fallback");
        Console.WriteLine($"PASS: {Checks} configuration, instance precedence and saved layout checks");
    }
}
