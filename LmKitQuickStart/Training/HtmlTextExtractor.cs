using System.Text;
using HtmlAgilityPack;

namespace LmKitQuickStart.Training;

/// <summary>Shared utility for stripping HTML to plain text for training samples.</summary>
public static class HtmlTextExtractor
{
    private static readonly string[] TagsToRemove = ["script", "style", "nav", "footer", "header", "aside"];

    public static string Extract(string html)
    {
        var doc = new HtmlDocument();
        doc.LoadHtml(html);

        foreach (string tag in TagsToRemove)
        {
            var nodes = doc.DocumentNode.SelectNodes($"//{tag}");
            if (nodes is not null)
                foreach (var node in nodes) node.Remove();
        }

        var content = doc.DocumentNode.SelectSingleNode("//main")
                   ?? doc.DocumentNode.SelectSingleNode("//article")
                   ?? doc.DocumentNode.SelectSingleNode("//body");

        if (content is null) return string.Empty;

        var sb = new StringBuilder();
        foreach (var node in content.DescendantsAndSelf())
        {
            if (node.NodeType != HtmlNodeType.Text) continue;
            string text = HtmlEntity.DeEntitize(node.InnerText).Trim();
            if (!string.IsNullOrWhiteSpace(text)) sb.AppendLine(text);
        }

        return sb.ToString();
    }
}
