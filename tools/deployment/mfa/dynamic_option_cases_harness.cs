using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using MFAAvalonia.Extensions.MaaFW;
using MFAAvalonia.Helper;

// Rendering, language lookup, and persistence are observable seams. The complete
// production refresh helper and the unchanged production ComboBox method are compiled.
namespace MFAAvalonia.Extensions.MaaFW
{
    public partial class MaaInterface
    {
        public Dictionary<string,MaaInterfaceOption>? Option { get; set; }
        public partial class MaaInterfaceOptionCase
        {
            [JsonProperty("name")] public string? Name { get; set; }
            [JsonProperty("label")] public string? Label { get; set; }
            [JsonProperty("label_args")] public Dictionary<string, string>? LabelArgs { get; set; }
            [JsonProperty("replacement_case")] public string? ReplacementCase { get; set; }
            [JsonProperty("description")] public string? Description { get; set; }
            [JsonProperty("icon")] public string? Icon { get; set; }
            [JsonProperty("option")] public List<string>? Option { get; set; }
            [JsonProperty("pipeline_override")] public Dictionary<string, JToken>? PipelineOverride { get; set; }
            public string DisplayName { get; set; } = "";
            public string DisplayDescription { get; set; } = "";
            public bool HasDescription { get; set; }
            public int Subscriptions { get; private set; }
            public void InitializeDisplayName() { Subscriptions++; RefreshDisplayMetadata(); }
            private void UpdateIcon() { }
        }
        public class MaaInterfaceOption
        {
            [JsonProperty("dynamic_cases")] public bool DynamicCases { get; set; }
            [JsonProperty("type")] public string? Type { get; set; }
            [JsonProperty("cases")] public List<MaaInterfaceOptionCase>? Cases { get; set; }
            public bool IsSelect => (Type?.ToLower() ?? "select") == "select";
            [JsonIgnore] public List<string>? DefaultCases { get; set; }
            public string? DefaultCase => DefaultCases?.FirstOrDefault();
            [JsonProperty("default_case")] public JToken? DefaultJson
            {
                get => DefaultCases==null ? null : new JArray(DefaultCases);
                set => DefaultCases=value==null ? null : value.Type==JTokenType.String ? new(){value.Value<string>()!} : value.ToObject<List<string>>();
            }
            public string? Description { get; set; }
            public List<string>? Document { get; set; }
            public void InitializeIcon() { }
        }
        public class MaaInterfaceSelectOption
        {
            public string? Name { get; set; }
            public int? Index { get; set; }
            public string? DisplayName { get; set; }
            public List<MaaInterfaceSelectOption>? SubOptions { get; set; }
        }
    }
}
namespace MFAAvalonia.Helper
{
    public static class Instances { public static RootModel RootViewModel = new(); }
    public class RootModel { public bool Idle=true; }
    public static class MaaProcessor { public static MaaInterface? Interface; }
    public static class LanguageHelper
    {
        public static Dictionary<string, string> Translations = new();
        public static string GetLocalizedDisplayName(string? label, string name) =>
            label != null && Translations.TryGetValue(label, out var text) ? text : label ?? name;
        public static string GetLocalizedString(string? value) => value ?? "";
    }
    public static class ContentExtensions
    {
        public static Task<string> ResolveContentAsync(this string? value) => Task.FromResult(value ?? "");
    }
    public static class LoggerHelper { public static List<string> Warnings = new(); public static void Warning(string message) => Warnings.Add(message); }
    public static class AppPaths { public static string InterfaceJsonPath = ""; }
    public class Change { public object? Property; }
    public class Control
    {
        public static readonly object IsEnabledProperty = new();
        public Thickness Margin { get; set; } = new(0,0,0,0);
        public event EventHandler<Change>? PropertyChanged;
        private bool enabled = true;
        public bool IsEnabled { get => enabled; set { if (enabled == value) return; enabled = value; PropertyChanged?.Invoke(this, new Change { Property = IsEnabledProperty }); } }
        public void Bind(object property, object value) { }
    }
    public class StackPanel : Control { public List<Control> Children { get; } = new(); }
    public class Grid : StackPanel { public static void SetColumn(Control c, int n) { } }
    public class Thickness(int a,int b,int c,int d) { }
    public enum HorizontalAlignment { Stretch }
    public class ComboBox : Control
    {
        public double MinWidth { get; set; }
        public List<string> Classes { get; } = new();
        public HorizontalAlignment HorizontalAlignment { get; set; }
        private IList<MaaInterface.MaaInterfaceOptionCase>? items;
        private object? selected;
        public IList<MaaInterface.MaaInterfaceOptionCase>? ItemsSource
        {
            get => items;
            set { items = value; selected = null; SelectionChanged?.Invoke(this, EventArgs.Empty); }
        }
        public object? SelectedItem { get => selected; set { selected = value; SelectionChanged?.Invoke(this, EventArgs.Empty); } }
        public int SelectedIndex => selected is MaaInterface.MaaInterfaceOptionCase c ? items?.IndexOf(c) ?? -1 : -1;
        public event EventHandler? SelectionChanged;
        public event EventHandler? DropDownOpened;
        public void Open() => DropDownOpened?.Invoke(this, EventArgs.Empty);
    }
    public static class ComboBoxExtensions
    {
        public static object SearchWatermarkProperty = new();
        public static void SetDisableNavigationOnLostFocus(ComboBox b, bool v) { }
        public static void SetCanSearch(ComboBox b, bool v) { }
        public static void SetSearchMemberPath(ComboBox b, string v) { }
    }
    public class I18nBinding(string key) { }
    public static class LangKeys { public const string Search = "Search"; }
    public class DragItemViewModel
    {
        public bool IsResourceOptionItem;
        public ResourceItem? ResourceItem;
        public TaskItem? InterfaceItem;
    }
    public class ResourceItem { public List<MaaInterface.MaaInterfaceSelectOption>? SelectOptions; }
    public class TaskItem { public List<MaaInterface.MaaInterfaceSelectOption>? Option; }
    public partial class TaskOptionGenerator
    {
        private readonly Action saveConfigurationAction;
        public int SubOptionRenders;
        public TaskOptionGenerator(Action save) { saveConfigurationAction = save; }
        public ComboBox Create(MaaInterface.MaaInterfaceSelectOption selection, MaaInterface.MaaInterfaceOption definition)
        {
            var wrapper = (StackPanel)CreateComboBoxControl(selection, definition, new DragItemViewModel());
            return ((Grid)wrapper.Children[0]).Children.OfType<ComboBox>().Single();
        }
        private Grid CreateBaseGrid() => new();
        private void BindIdleEnabled(Control c) { }
        private void SetupComboBoxTemplate(ComboBox b) { }
        private MaaInterface.MaaInterfaceSelectOption CreateDefaultSelectOption(string name) => new() { Name = name };
        private void AddSubOption(StackPanel c, MaaInterface.MaaInterfaceSelectOption o, DragItemViewModel s) { SubOptionRenders++; }
        private StackPanel CreateLabelPanel(string? name,string? id,string? description,List<string>? doc) => new();
        private Control CreateIcon(MaaInterface.MaaInterfaceOption o) => new();
        private void AddResponsiveBehavior(Grid grid,Control label,Control input) { }
        private Control CreateNestedOptionsBorder(StackPanel panel) => panel;
    }
}
namespace MFAAvalonia.ViewModels.Pages
{
    public partial class TaskQueueViewModel
    {
        public bool IsRunning;
        public List<DragItemViewModel> TaskItemViewModels = new();
        public int Saves;
        private void PersistConfigurationState() { Saves++; }
        public void PrepareStart() => RefreshDynamicSelectionsBeforeStart();
    }
}
public static class Program
{
    private static int passed;
    private static string root = "";
    private static void Check(bool condition, string label)
    {
        if (!condition) throw new Exception(label);
    }
    private static void Case(string label, Action body)
    {
        body(); passed++; Console.WriteLine("PASS " + label);
    }
    private static JObject Document()
    {
        return JObject.Parse("""{"option":{"season":{"type":"select","dynamic_cases":true,"cases":[{"name":"latest","label":"最新（第51期）","description":"预览","pipeline_override":{"Config":{"season":"latest"}}},{"name":"season-50","label":"第50期","option":["nested"],"pipeline_override":{"Config":{"season":50}}},{"name":"season-49","label":"第49期","pipeline_override":{"Config":{"season":49}}}]}}}""");
    }
    private static JArray Cases(JObject document) => (JArray)document["option"]!["season"]!["cases"]!;
    private static void Append(JObject document, int number)
    {
        Cases(document).Add(new JObject { ["name"] = "season-" + number, ["label"] = "第" + number + "期",
            ["pipeline_override"] = new JObject { ["Config"] = new JObject { ["season"] = number } } });
        Cases(document)[0]["label"] = "最新（第" + number + "期）";
        ((JObject)Cases(document)[0]).Remove("description");
    }
    private static MaaInterface.MaaInterfaceOption Model(JObject doc) => doc["option"]!["season"]!.ToObject<MaaInterface.MaaInterfaceOption>()!;
    private static void Write(JObject document) => File.WriteAllText(AppPaths.InterfaceJsonPath, document.ToString());
    private static string Name(ComboBox box) => ((MaaInterface.MaaInterfaceOptionCase)box.SelectedItem!).Name!;
    public static int Main(string[] args)
    {
        try
        {
            root=args[0]; Directory.CreateDirectory(root); AppPaths.InterfaceJsonPath=Path.Combine(root,"interface.json");
            Run();
            LocalizedLabels(args[1], args[2]);
            Replacements();
            if (args.Length == 5) RealProjection(args[3], args[4]);
            Console.WriteLine($"ALL {passed} CASES PASSED"); return 0;
        }
        catch(Exception error) { Console.Error.WriteLine(error); return 1; }
    }
    private static void Run()
    {
        Case("opt-out performs no file read and preserves source order", () => {
            var model=Model(Document()); model.DynamicCases=false;
            var warnings=LoggerHelper.Warnings.Count;
            Check(!DynamicOptionCases.TryRefresh("season",model,root), "disabled");
            Check(LoggerHelper.Warnings.Count==warnings,"no read failure");
            Check(ReferenceEquals(DynamicOptionCases.DisplayCases(model),model.Cases),"same list");
        });
        Case("default boolean is opt-out", () => {
            var doc=Document(); ((JObject)doc["option"]!["season"]!).Remove("dynamic_cases");
            Check(!Model(doc).DynamicCases,"default");
        });
        Case("startup keeps stable index despite display sorting and appends latest", () => {
            var doc=Document(); var model=Model(doc); Append(doc,51); Append(doc,52); Write(doc);
            var selection=new MaaInterface.MaaInterfaceSelectOption { Name="season",Index=1 };
            var saves=0; var box=new TaskOptionGenerator(()=>saves++).Create(selection,model);
            Check(model.Cases!.Select(c=>c.Name).SequenceEqual(new[]{"latest","season-50","season-49","season-51","season-52"}),"append order");
            Check(box.ItemsSource!.Select(c=>c.Name).SequenceEqual(new[]{"latest","season-52","season-51","season-50","season-49"}),"display order");
            Check(Name(box)=="season-50"&&box.SelectedIndex==3&&selection.Index==1&&saves==0,"stable identity");
            Check(model.Cases[0].DisplayName=="最新（第52期）"&&!model.Cases[0].HasDescription,"metadata refreshed");
        });
        Case("dropdown refresh preserves selection sub-options and language subscription", () => {
            var doc=Document(); Write(doc); var model=Model(doc);
            var nested=new MaaInterface.MaaInterfaceSelectOption { Name="nested",Index=7 };
            var selection=new MaaInterface.MaaInterfaceSelectOption { Name="season",Index=1,SubOptions=new(){nested} };
            var saves=0; var generator=new TaskOptionGenerator(()=>saves++); var box=generator.Create(selection,model);
            var renders=generator.SubOptionRenders; Append(doc,52); Write(doc); box.Open(); box.Open();
            Check(Name(box)=="season-50"&&selection.Index==1&&saves==0,"no refresh save");
            Check(ReferenceEquals(selection.SubOptions![0],nested)&&nested.Index==7&&generator.SubOptionRenders==renders,"no sub-option reconstruction");
            Check(model.Cases!.All(c=>c.Subscriptions==1),"no duplicate subscriptions");
            box.SelectedItem=box.ItemsSource!.Single(c=>c.Name=="season-52");
            Check(selection.Index==3&&saves==1,"selection maps name to persisted index");
        });
        Case("idle re-enable sees new cases without resuming a task", () => {
            var doc=Document(); Write(doc); var model=Model(doc);
            var selection=new MaaInterface.MaaInterfaceSelectOption { Name="season",Index=0 };
            var saves=0; var box=new TaskOptionGenerator(()=>saves++).Create(selection,model);
            box.IsEnabled=false; Append(doc,52); Write(doc);
            Check(model.Cases!.Count==3,"disabled did not refresh");
            box.IsEnabled=true;
            Check(model.Cases.Count==4&&Name(box)=="latest"&&saves==0,"idle refresh only");
        });
        Case("another dropdown refreshes its display after shared model changed", () => {
            var doc=Document(); Write(doc); var model=Model(doc); var saves=0;
            var generator=new TaskOptionGenerator(()=>saves++);
            var first=generator.Create(new(){Name="season",Index=0},model);
            var second=generator.Create(new(){Name="season",Index=2},model);
            Append(doc,52); Write(doc); first.Open(); second.Open();
            Check(first.ItemsSource!.Count==4&&second.ItemsSource!.Count==4&&Name(second)=="season-49"&&saves==0,"shared option reflected");
        });
        Case("restart restores a newly appended numeric season by stable index", () => {
            var doc=Document(); Append(doc,51); Append(doc,52); Write(doc);
            var model=Model(doc); var selection=new MaaInterface.MaaInterfaceSelectOption {Name="season",Index=4};
            var saves=0; var box=new TaskOptionGenerator(()=>saves++).Create(selection,model);
            Check(Name(box)=="season-52"&&box.SelectedIndex==1&&selection.Index==4&&saves==0,"restart");
            Check(model.Cases![selection.Index!.Value].PipelineOverride!["Config"]["season"]!.Value<int>()==52,"pipeline stays 52");
        });
        var invalid=new Dictionary<string,Action<JObject>> {
            ["reordered prefix"]=doc=>{var a=Cases(doc)[1].DeepClone();var b=Cases(doc)[2].DeepClone();Cases(doc)[1]=b;Cases(doc)[2]=a;},
            ["deleted prefix"]=doc=>Cases(doc).RemoveAt(1),
            ["pipeline tamper"]=doc=>Cases(doc)[1]["pipeline_override"]!["Config"]!["season"]=99,
            ["sub-option tamper"]=doc=>Cases(doc)[1]["option"]=new JArray("other"),
            ["icon tamper"]=doc=>Cases(doc)[1]["icon"]="other.png",
            ["duplicate name"]=doc=>Cases(doc).Add(Cases(doc)[1].DeepClone()),
            ["null entry"]=doc=>Cases(doc).Add(JValue.CreateNull()),
            ["blank name"]=doc=>Cases(doc).Add(new JObject {["name"]=" "}),
        };
        foreach(var (label,alter) in invalid) Case(label+" rejects the whole update",()=>{
            var doc=Document(); var model=Model(doc); Append(doc,52); alter(doc); Write(doc);
            Check(!DynamicOptionCases.TryRefresh("season",model,AppPaths.InterfaceJsonPath),"rejected");
            Check(model.Cases!.Count==3&&model.Cases[0].Label=="最新（第51期）","atomic rejection");
        });
        Case("truncated file and missing file retain the current model",()=>{
            var model=Model(Document()); File.WriteAllText(AppPaths.InterfaceJsonPath,"{\"option\":");
            Check(!DynamicOptionCases.TryRefresh("season",model,AppPaths.InterfaceJsonPath)&&model.Cases!.Count==3,"truncated");
            Check(!DynamicOptionCases.TryRefresh("season",model,Path.Combine(root,"missing")),"missing");
        });
        Case("same-name source opt-in and select type are required",()=>{
            var doc=Document(); Append(doc,52); var model=Model(Document()); Write(doc);
            Check(!DynamicOptionCases.TryRefresh("unrelated",model,AppPaths.InterfaceJsonPath),"exact name");
            doc["option"]!["season"]!["dynamic_cases"]=false; Write(doc);
            Check(!DynamicOptionCases.TryRefresh("season",model,AppPaths.InterfaceJsonPath),"source opt-out");
            doc["option"]!["season"]!["dynamic_cases"]=true; doc["option"]!["season"]!["type"]="input"; Write(doc);
            Check(!DynamicOptionCases.TryRefresh("season",model,AppPaths.InterfaceJsonPath),"source type");
            model.Type="checkbox";
            Check(!DynamicOptionCases.TryRefresh("season",model,root),"model type");
        });
        Case("label-only update refreshes existing objects without changing pipeline",()=>{
            var doc=Document(); var model=Model(doc); var original=model.Cases![0];
            Cases(doc)[0]["label"]="最新（第52期）"; ((JObject)Cases(doc)[0]).Remove("description"); Write(doc);
            Check(DynamicOptionCases.TryRefresh("season",model,AppPaths.InterfaceJsonPath),"display-only changed");
            Check(ReferenceEquals(original,model.Cases[0])&&!original.HasDescription&&original.PipelineOverride!["Config"]["season"]!.Value<string>()=="latest","same symbolic latest");
            Check(!DynamicOptionCases.TryRefresh("season",model,AppPaths.InterfaceJsonPath),"idempotent");
        });
        Case("description-only update changes preview presentation",()=>{
            var doc=Document(); var model=Model(doc);
            ((JObject)Cases(doc)[0]).Remove("description"); Write(doc);
            Check(DynamicOptionCases.TryRefresh("season",model,AppPaths.InterfaceJsonPath),"description changed");
            Check(model.Cases![0].Description==null&&model.Cases[0].Label=="最新（第51期）","label retained");
        });
        Case("invalid option shapes leave the model unchanged",()=>{
            var model=Model(Document());
            foreach (var shape in new[]{"{\"option\":true}","{\"option\":{\"season\":42}}","{}"})
            {
                File.WriteAllText(AppPaths.InterfaceJsonPath,shape);
                Check(!DynamicOptionCases.TryRefresh("season",model,AppPaths.InterfaceJsonPath),"invalid shape rejected");
            }
            Check(model.Cases!.Count==3,"unchanged");
        });
        Case("ordinary select never gains dynamic event handlers",()=>{
            var doc=Document(); Write(doc); var model=Model(doc); model.DynamicCases=false; var saves=0;
            var box=new TaskOptionGenerator(()=>saves++).Create(new(){Name="season",Index=1},model);
            Append(doc,52); Write(doc); box.Open(); box.IsEnabled=false; box.IsEnabled=true;
            Check(model.Cases!.Count==3&&saves==0&&Name(box)=="season-50","ordinary unchanged");
        });
    }
    private static void RealProjection(string beforePath, string afterPath)
    {
        Case("real Python projection preserves all historical indices and displayed selection",()=>{
            var before=JObject.Parse(File.ReadAllText(beforePath));
            var after=JObject.Parse(File.ReadAllText(afterPath));
            var model=before["option"]!["竞技场期数"]!.ToObject<MaaInterface.MaaInterfaceOption>()!;
            model.DynamicCases=true;
            var names=model.Cases!.Select(c=>c.Name).ToArray();
            var oldPipelines=model.Cases.Select(c=>JToken.FromObject(c.PipelineOverride!)).ToArray();
            File.Copy(afterPath,AppPaths.InterfaceJsonPath,true);
            var saves=0; var selection=new MaaInterface.MaaInterfaceSelectOption {Name="竞技场期数",Index=1};
            var box=new TaskOptionGenerator(()=>saves++).Create(selection,model);
            var expected=(JArray)after["option"]!["竞技场期数"]!["cases"]!;
            Check(model.Cases.Count==expected.Count,"all projected cases received");
            for(var i=0;i<names.Length;i++)
                Check(model.Cases[i].Name==names[i]&&JToken.DeepEquals(JToken.FromObject(model.Cases[i].PipelineOverride!),oldPipelines[i]),"historical index "+i);
            Check(Name(box)==names[1]&&selection.Index==1&&saves==0,"fixed old season retained");
            Check(model.Cases[0].Label==expected[0]["label"]!.Value<string>(),"latest presentation follows provided definition");
            var finalName=model.Cases.Last().Name;
            box.SelectedItem=box.ItemsSource!.Single(c=>c.Name==finalName);
            Check(selection.Index==model.Cases.Count-1&&saves==1,"appended case saves stable index");
        });
    }
    private static void LocalizedLabels(string cnPath, string hantPath)
    {
        foreach(var (path,previewText) in new[]{(cnPath,"预览"),(hantPath,"預覽")})
        Case("actual localized templates and label arguments " + Path.GetFileName(path),()=>{
            LanguageHelper.Translations=JObject.Parse(File.ReadAllText(path)).Properties()
                .ToDictionary(p=>"$"+p.Name,p=>p.Value.Value<string>()!);
            var doc=Document();
            Cases(doc)[0]["label"]="$竞技场最新期格式";
            Cases(doc)[0]["label_args"]=new JObject { ["season"]="51" };
            var model=Model(doc); Write(doc);
            var saves=0; var box=new TaskOptionGenerator(()=>saves++).Create(new(){Name="season",Index=0},model);
            Check(model.Cases![0].DisplayName=="最新（第 51 期）","initial localized season");
            Cases(doc)[0]["label_args"]!["season"]="52"; Write(doc); box.Open();
            Check(model.Cases[0].DisplayName=="最新（第 52 期）","arguments-only refresh");
            Append(doc,52);
            Cases(doc)[0]["label"]="$竞技场最新预览期格式";
            Cases(doc).Last!["label"]="$竞技场固定预览期格式";
            Cases(doc).Last!["label_args"]=new JObject { ["season"]="52" };
            Cases(doc)[1]["label"]="$竞技场最近正式期格式";
            Cases(doc)[1]["label_args"]=new JObject { ["season"]="50" };
            Write(doc); box.Open();
            Check(model.Cases[0].DisplayName=="最新（第 52 期，"+previewText+"）","same latest id changes to preview");
            Check(model.Cases.Last().DisplayName=="第 52 期（"+previewText+"）","new preview case localized");
            Check(model.Cases[1].DisplayName=="第 50 期（最近正式）","prior formal case localized");
            Check(Name(box)=="latest"&&saves==0&&model.Cases[0].Subscriptions==1,"no selection or subscription changes");
            Check(model.Cases[0].PipelineOverride!["Config"]["season"]!.Value<string>()=="latest","latest symbolic behavior retained");
        });
        Case("label substitution preserves unknown tokens and never expands argument values",()=>{
            var item=new MaaInterface.MaaInterfaceOptionCase {
                Name="literal",Label="{season}/{unknown}",LabelArgs=new(){["season"]="{unknown}",["unused"]="x"}
            };
            item.InitializeDisplayName();
            Check(item.DisplayName=="{unknown}/{unknown}","single substitution pass");
        });
        LanguageHelper.Translations.Clear();
    }
    private static JObject ReplacementDocument(int latest)
    {
        var doc=Document(); Append(doc,51); Append(doc,52);
        if(latest>52)Append(doc,latest);
        Cases(doc)[0]["replacement_case"]="season-"+latest;
        doc["option"]!["season"]!["default_case"]="season-"+latest;
        return doc;
    }
    private static void Replacements()
    {
        Case("compatibility alias is invisible and migrates only once",()=>{
            var original=Model(Document()); var doc=ReplacementDocument(52); Write(doc);
            var selection=new MaaInterface.MaaInterfaceSelectOption{Name="season",Index=0};
            var saves=0;var box=new TaskOptionGenerator(()=>saves++).Create(selection,original);
            Check(box.ItemsSource!.All(c=>c.Name!="latest")&&box.ItemsSource.Count==4,"alias hidden");
            Check(selection.Index==4&&Name(box)=="season-52"&&saves==1,"one-time migration saved");
            Write(ReplacementDocument(53));box.Open();box.Open();
            Check(selection.Index==4&&Name(box)=="season-52"&&saves==1,"selected period never follows new alias target");
            Check(original.DefaultCase=="season-53","new default updated");
        });
        Case("unopened options migrate and save before starting",()=>{
            var model=Model(Document());Write(ReplacementDocument(52));
            MaaProcessor.Interface=new(){Option=new(){["season"]=model}};
            var old=new MaaInterface.MaaInterfaceSelectOption{Name="season",Index=0};
            var queue=new MFAAvalonia.ViewModels.Pages.TaskQueueViewModel();
            queue.TaskItemViewModels.Add(new(){InterfaceItem=new(){Option=new(){old}}});
            queue.PrepareStart();queue.PrepareStart();
            Check(old.Index==4&&queue.Saves==1,"direct start migration persists exactly once");
            Write(ReplacementDocument(53));queue.PrepareStart();
            Check(old.Index==4&&queue.Saves==1,"next start keeps fixed selection");
        });
        Case("running tasks neither refresh definitions nor migrate selection",()=>{
            var model=Model(Document());Write(ReplacementDocument(52));
            MaaProcessor.Interface=new(){Option=new(){["season"]=model}};
            var old=new MaaInterface.MaaInterfaceSelectOption{Name="season",Index=0};
            var queue=new MFAAvalonia.ViewModels.Pages.TaskQueueViewModel {IsRunning=true};
            queue.TaskItemViewModels.Add(new(){InterfaceItem=new(){Option=new(){old}}});
            queue.PrepareStart();queue.IsRunning=false;Instances.RootViewModel.Idle=false;queue.PrepareStart();
            Check(old.Index==0&&model.Cases!.Count==3&&queue.Saves==0,"active state unchanged");
            Instances.RootViewModel.Idle=true;queue.PrepareStart();
            Check(old.Index==4&&queue.Saves==1,"idle migration resumes");
        });
        Case("fixed choice and nested resource choices use the same resolver",()=>{
            var model=Model(ReplacementDocument(52));Write(ReplacementDocument(53));
            var fixedChoice=new MaaInterface.MaaInterfaceSelectOption{Name="season",Index=3};
            var nested=new MaaInterface.MaaInterfaceSelectOption{Name="season",Index=0};
            var parent=new MaaInterface.MaaInterfaceSelectOption{Name="parent",SubOptions=new(){nested}};
            Check(DynamicOptionCases.RefreshSelections(new[]{fixedChoice,parent},new(){["season"]=model},AppPaths.InterfaceJsonPath),"nested migration");
            Check(fixedChoice.Index==3&&nested.Index==5,"fixed 51 retained and old alias resolved");
            var fresh=new MaaInterface.MaaInterfaceSelectOption{Name="season"};
            Check(DynamicOptionCases.ResolveSelection(fresh,model)&&fresh.Index==5,"default resolves to concrete new index");
        });
        foreach(var kind in new[]{"missing","self","chain","default-alias"})
        Case("invalid replacement update rejected atomically: "+kind,()=>{
            var model=Model(Document());var doc=ReplacementDocument(52);
            if(kind=="missing")Cases(doc)[0]["replacement_case"]="missing";
            if(kind=="self")Cases(doc)[0]["replacement_case"]="latest";
            if(kind=="chain")Cases(doc)[4]["replacement_case"]="season-51";
            if(kind=="default-alias")doc["option"]!["season"]!["default_case"]="latest";
            Write(doc);Check(!DynamicOptionCases.TryRefresh("season",model,AppPaths.InterfaceJsonPath),"rejected");
            Check(model.Cases!.Count==3&&model.Cases[0].ReplacementCase==null,"old definition preserved");
        });
    }
}
