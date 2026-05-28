using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using FarPoint.Win.Spread;
using FarPoint.Win.Spread.CellType;

[Serializable]
public class ChatBubbleCellType : TextCellType
{
    public bool IsMine { get; set; }

    public Color MyBubbleColor { get; set; } = Color.FromArgb(255, 235, 120);
    public Color OtherBubbleColor { get; set; } = Color.FromArgb(240, 240, 240);
    public Color TextColor { get; set; } = Color.Black;

    public int CornerRadius { get; set; } = 12;
    public int TailWidth { get; set; } = 10;
    public int TailHeight { get; set; } = 12;

    public override void PaintCell(
        Graphics g,
        Rectangle r,
        Appearance appearance,
        object value,
        bool isSelected,
        bool isLocked,
        float zoomFactor)
    {
        g.SmoothingMode = SmoothingMode.AntiAlias;

        string text = Convert.ToString(value);
        Color bubbleColor = IsMine ? MyBubbleColor : OtherBubbleColor;

        Rectangle bubbleRect = r;
        bubbleRect.Inflate(-8, -6);

        // 꼬리가 들어갈 공간 확보
        if (IsMine)
        {
            bubbleRect.X += 18;
            bubbleRect.Width -= TailWidth + 18;
        }
        else
        {
            bubbleRect.X += TailWidth;
            bubbleRect.Width -= TailWidth + 18;
        }

        using (GraphicsPath bubblePath = CreateBubblePath(bubbleRect, IsMine))
        using (SolidBrush bubbleBrush = new SolidBrush(bubbleColor))
        {
            g.FillPath(bubbleBrush, bubblePath);
        }

        Rectangle textRect = bubbleRect;
        textRect.Inflate(-10, -6);

        TextRenderer.DrawText(
            g,
            text,
            appearance.Font,
            textRect,
            TextColor,
            TextFormatFlags.WordBreak |
            TextFormatFlags.Left |
            TextFormatFlags.VerticalCenter |
            TextFormatFlags.NoPrefix
        );
    }

    private GraphicsPath CreateBubblePath(Rectangle rect, bool isMine)
    {
        int radius = CornerRadius;
        int diameter = radius * 2;

        int tailMiddleY = rect.Top + 18;
        int tailTopY = tailMiddleY - TailHeight / 2;
        int tailBottomY = tailMiddleY + TailHeight / 2;

        GraphicsPath path = new GraphicsPath();

        if (isMine)
        {
            // 오른쪽 꼬리 말풍선

            path.AddArc(rect.X, rect.Y, diameter, diameter, 180, 90);
            path.AddLine(rect.X + radius, rect.Y, rect.Right - radius, rect.Y);
            path.AddArc(rect.Right - diameter, rect.Y, diameter, diameter, 270, 90);

            // 오른쪽 상단 이후 꼬리
            path.AddLine(rect.Right, rect.Y + radius, rect.Right, tailTopY);
            path.AddLine(rect.Right, tailTopY, rect.Right + TailWidth, tailMiddleY);
            path.AddLine(rect.Right + TailWidth, tailMiddleY, rect.Right, tailBottomY);

            path.AddLine(rect.Right, tailBottomY, rect.Right, rect.Bottom - radius);
            path.AddArc(rect.Right - diameter, rect.Bottom - diameter, diameter, diameter, 0, 90);
            path.AddLine(rect.Right - radius, rect.Bottom, rect.X + radius, rect.Bottom);
            path.AddArc(rect.X, rect.Bottom - diameter, diameter, diameter, 90, 90);
            path.AddLine(rect.X, rect.Bottom - radius, rect.X, rect.Y + radius);
        }
        else
        {
            // 왼쪽 꼬리 말풍선

            path.AddArc(rect.X, rect.Y, diameter, diameter, 180, 90);
            path.AddLine(rect.X + radius, rect.Y, rect.Right - radius, rect.Y);
            path.AddArc(rect.Right - diameter, rect.Y, diameter, diameter, 270, 90);
            path.AddLine(rect.Right, rect.Y + radius, rect.Right, rect.Bottom - radius);
            path.AddArc(rect.Right - diameter, rect.Bottom - diameter, diameter, diameter, 0, 90);
            path.AddLine(rect.Right - radius, rect.Bottom, rect.X + radius, rect.Bottom);
            path.AddArc(rect.X, rect.Bottom - diameter, diameter, diameter, 90, 90);

            // 왼쪽 하단 이후 꼬리
            path.AddLine(rect.X, rect.Bottom - radius, rect.X, tailBottomY);
            path.AddLine(rect.X, tailBottomY, rect.X - TailWidth, tailMiddleY);
            path.AddLine(rect.X - TailWidth, tailMiddleY, rect.X, tailTopY);

            path.AddLine(rect.X, tailTopY, rect.X, rect.Y + radius);
        }

        path.CloseFigure();
        return path;
    }
}