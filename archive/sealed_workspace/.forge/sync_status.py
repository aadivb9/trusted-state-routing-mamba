"""Generate current-status blocks from saved results; never edit experiment evidence."""
import re
from check import ROOT, ENTRY_POINTS, START, END, current_status


def main():
    for name in ENTRY_POINTS:
        path = ROOT / name
        text = path.read_text()
        block = current_status(ROOT, name)
        if START in text or END in text:
            if text.count(START) != 1 or text.count(END) != 1:
                raise ValueError(f'Malformed status markers in {name}')
            text = re.sub(re.escape(START) + r'.*?' + re.escape(END), lambda _: block, text, flags=re.S)
        else:
            title, rest = text.split('\n', 1)
            text = title + '\n\n' + block + '\n' + rest
        path.write_text(text)


if __name__ == '__main__':
    main()
