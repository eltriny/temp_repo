public class ChatBubbleCellType : FarPoint.Win.Spread.CellType.TextCellType
{
    public bool IsMine { get; set; }

    public override void PaintCell(
        Graphics g,
        Rectangle r,
        FarPoint.Win.Spread.Appearance appearance,
        object value,
        bool isSelected,
        bool isLocked,
        float zoomFactor)
    {
        Color bubbleColor = IsMine
            ? Color.FromArgb(255, 235, 120)
            : Color.FromArgb(240, 240, 240);

        Rectangle bubbleRect = r;
        bubbleRect.Inflate(-8, -6);

        if (IsMine)
        {
            bubbleRect.X += 20;
            bubbleRect.Width -= 20;
        }
        else
        {
            bubbleRect.Width -= 20;
        }

        using (GraphicsPath path = CreateRoundRectPath(bubbleRect, 12))
        using (SolidBrush brush = new SolidBrush(bubbleColor))
        {
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            g.FillPath(brush, path);
        }

        Rectangle textRect = bubbleRect;
        textRect.Inflate(-10, -6);

        TextRenderer.DrawText(
            g,
            Convert.ToString(value),
            appearance.Font,
            textRect,
            Color.Black,
            TextFormatFlags.WordBreak |
            TextFormatFlags.VerticalCenter |
            TextFormatFlags.Left
        );
    }

    private GraphicsPath CreateRoundRectPath(Rectangle rect, int radius)
    {
        int diameter = radius * 2;
        GraphicsPath path = new GraphicsPath();

        path.AddArc(rect.X, rect.Y, diameter, diameter, 180, 90);
        path.AddArc(rect.Right - diameter, rect.Y, diameter, diameter, 270, 90);
        path.AddArc(rect.Right - diameter, rect.Bottom - diameter, diameter, diameter, 0, 90);
        path.AddArc(rect.X, rect.Bottom - diameter, diameter, diameter, 90, 90);

        path.CloseFigure();
        return path;
    }
}