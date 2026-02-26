from PIL import Image, ImageDraw, ImageFont

def wrap_text(text, max_width, draw, font):
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        line_width = draw.textlength(" ".join(current_line), font=font)

        if line_width > max_width:
            if len(current_line) == 1:
                lines.append(current_line[0])
                current_line = []
            else:
                current_line.pop()
                lines.append(" ".join(current_line))
                current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines

def test_wrap_text():
    img = Image.new("RGB", (100, 100))
    draw = ImageDraw.Draw(img)
    # Use default font or load a specific one if available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except IOError:
        font = ImageFont.load_default()

    text = "This is a long text that should be wrapped"
    max_width = 50

    lines = wrap_text(text, max_width, draw, font)
    print(f"Wrapped lines: {lines}")
    assert len(lines) > 1, "Should have wrapped to multiple lines"

    for line in lines:
        length = draw.textlength(line, font=font)
        print(f"Line: '{line}', Length: {length}")
        # Note: if a single word is longer than max_width, it will be kept as is.
        # But here words are short.

if __name__ == "__main__":
    test_wrap_text()
    print("Test passed!")
