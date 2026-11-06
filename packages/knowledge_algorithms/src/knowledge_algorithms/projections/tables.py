"""从转换器规范 Markdown 保留可定位表头与完整行；不推测跨页合并或缺失单元格。"""

import re


def markdown_table_spans(text):
    lines, offset = [], 0
    for line in text.splitlines(keepends=True):
        value = line.rstrip("\r\n")
        if value.strip():
            lines.append((offset, offset + len(value), value))
        offset += len(line)
    if len(lines) < 3:
        return None, ()
    def cells(value):
        # 含转义分隔符的表格交给明确原生单元格适配器，不猜测列数。
        if r"\|" in value:
            return ()
        return tuple(cell.strip() for cell in value.strip().strip("|").split("|"))
    header, separator = cells(lines[0][2]), cells(lines[1][2])
    if len(header) < 2 or len(header) != len(separator) or not all(
            re.fullmatch(r":?-{3,}:?", value) for value in separator):
        return None, ()
    if not all(len(cells(line[2])) == len(header) for line in lines[2:]):
        return None, ()
    return (lines[0][0], lines[1][1]), tuple((line[0], line[1]) for line in lines[2:])
