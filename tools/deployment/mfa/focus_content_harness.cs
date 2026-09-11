using System;
using System.IO;
using System.Collections.Generic;

static class LoggerHelper
{
    public static readonly List<string> Warnings = new();
    public static void Warning(string message) => Warnings.Add(message);
}

static class MaaInterface
{
    public static string ReplacePlaceholder(string text, string root, bool _) => Path.Combine(root, text);
}

partial class FocusHandler
{
    static int checks;
    static void Check(string text, string resolved, string root, bool warning)
    {
        LoggerHelper.Warnings.Clear();
        LogUnresolvedFocusPath(text, resolved, root);
        if (LoggerHelper.Warnings.Count != (warning ? 1 : 0))
            throw new Exception("Unexpected file warning: " + text);
        checks++;
    }

    public static void Main(string[] args)
    {
        string root = args[0];
        foreach (string text in new[] {
            "[color:orange]挑战准备期间[/color]",
            "[color:LimeGreen]奖励已领取[/color]",
            "[COLOR:Orange]第一行\n第二行[/COLOR]",
            "完成：[color:green]1/5[/color]。",
            "普通文本", "", "https://example.invalid/notice.md", "$daily.finished"
        }) Check(text, text, root, false);
        foreach (string text in new[] {
            "./missing.md", "../help.txt", @"C:\missing\help.md",
            "resource/notice.md", @"resource\notice.md", "help.md"
        }) Check(text, text, root, true);
        Check("resource/notice.md", "已读取的正文", root, false);
        File.WriteAllText(Path.Combine(root, "same.txt"), "same.txt");
        Check("same.txt", "same.txt", root, true);
        if (!LoggerHelper.Warnings[0].Contains("exists=True"))
            throw new Exception("Existing-file diagnostic lost");
        checks++;
        Console.WriteLine($"PASS: {checks} focus text and file diagnostic checks");
    }
}
