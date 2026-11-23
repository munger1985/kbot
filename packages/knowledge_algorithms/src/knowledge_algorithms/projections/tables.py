"""从转换器规范 Markdown 保留可定位表头与完整行；不推测跨页合并或缺失单元格。"""

import re


def markdown_cell_spans(text, span):
    """保留转义与代码内的竖线；不能确认单行语法时不返回列。"""
    start, stop = span
    value = text[start:stop]
    delimiters = []
    code = 0
    for token in re.finditer(r"\\.|[\x60]+|\||\r|\n", value):
        symbol = token[0]
        if symbol in {"\r", "\n"}:
            return ()
        if symbol.startswith(chr(96)):
            if not code:
                code = len(symbol)
            elif code == len(symbol):
                code = 0
        elif symbol == "|" and not code:
            delimiters.append(start + token.start())
    if code or not delimiters:
        return ()
    boundaries = [start, *delimiters, stop]
    cells = tuple((left, right) for left, right in zip(boundaries, boundaries[1:])
                  if left < right)
    if cells and not text[slice(*cells[0])].strip():
        cells = cells[1:]
    if cells and text[slice(*cells[-1])].strip() == "|":
        cells = cells[:-1]
    return cells


def markdown_table_spans(text):
    lines, offset = [], 0
    for line in text.splitlines(keepends=True):
        value = line.rstrip("\r\n")
        if value.strip():
            lines.append((offset, offset + len(value), value))
        offset += len(line)
    if len(lines) < 3:
        return None, ()
    def cells(line):
        return tuple(text[a:b].removeprefix("|").strip() for a, b in markdown_cell_spans(text, line[:2]))
    # Docling 的原生表格序列化可能带图注/前言，也可能输出紧凑分隔行。
    candidates = []
    for index in range(1, len(lines) - 1):
        separator = cells(lines[index])
        if separator and all(re.fullmatch(r":?-+:?", value) for value in separator):
            header = cells(lines[index - 1])
            if len(header) == len(separator) and any(header):
                candidates.append((index, len(header)))
    if len(candidates) != 1:
        return None, ()
    index, width = candidates[0]
    if not all(len(cells(line)) == width for line in lines[index + 1:]):
        return None, ()
    return (lines[index - 1][0], lines[index][1]), tuple((line[0], line[1]) for line in lines[index + 1:])
