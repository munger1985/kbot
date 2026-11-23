// Provider 负责决定字段是否可填；此处只定义表单控件与中文名称。
const fields = Object.freeze({
  max_tokens: ["最大输出 Token 数", "integer", 1],
  temperature: ["采样温度", "number"],
  top_p: ["采样范围（Top P）", "number"],
  frequency_penalty: ["频次惩罚", "number"],
  presence_penalty: ["出现惩罚", "number"],
  timeout: ["请求超时（秒）", "integer", 1],
  timeout_seconds: ["请求超时（秒）", "number", 0],
  max_retries: ["最大重试次数", "integer", 0],
  compartment_id: ["OCI Compartment OCID", "text"],
  project: ["OCI Generative AI Project OCID", "text"],
  model_revision: ["模型版本", "text"],
  processor_revision: ["预处理器版本", "text"],
  input_template: ["输入模板", "textarea"],
  embedding_dimension: ["文本向量维度", "integer", 1],
  dimension: ["视觉向量维度", "integer", 1],
  model_path: ["模型权重路径", "text"],
  device: ["运行设备", "text"],
  batch_size: ["批大小", "integer", 1],
  normalize: ["向量归一化", "boolean"],
  retrieval_directions: ["检索方向", "directions"],
  max_representation_elements: ["单条表示最大元素数", "integer", 1],
  languages: ["识别语言代码", "list"],
  num_threads: ["线程数", "integer", 1],
  config_path: ["RapidOCR 配置文件路径", "text"],
  rapidocr_params: ["RapidOCR 参数覆盖", "map"],
  download_enabled: ["允许下载模型", "boolean"],
  executable: ["Tesseract 可执行文件路径", "text"],
  prompt: ["识别提示词", "textarea"],
});

const directionOptions = [
  ["TEXT_TO_IMAGE", "以文搜图"], ["IMAGE_TO_IMAGE", "以图搜图"],
  ["IMAGE_TO_TEXT", "以图搜文"], ["TEXT_TO_TEXT", "以文搜文"],
];
const ociOnly = new Set(["compartment_id", "project"]);

function node(tag, className = "", copy = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (copy) element.textContent = copy;
  return element;
}

function choices(options) {
  const select = node("select");
  select.replaceChildren(...options.map(([value, name]) => new Option(name, value)));
  return select;
}

function scalar(kind, value, min) {
  if (kind === "boolean") {
    const control = choices([["", "使用默认值"], ["true", "是"], ["false", "否"]]);
    control.value = typeof value === "boolean" ? String(value) : "";
    return [control, () => control.value === "" ? undefined : control.value === "true"];
  }
  const control = node(kind === "textarea" ? "textarea" : "input");
  if (["number", "integer"].includes(kind)) {
    control.type = "number";
    control.step = kind === "integer" ? "1" : "any";
    if (min !== undefined) control.min = String(min);
    control.value = value ?? "";
    return [control, () => control.value === "" ? undefined : control.valueAsNumber];
  }
  if (kind === "list") {
    control.placeholder = "例如 ch_sim, en";
    control.value = Array.isArray(value) ? value.join(", ") : (value ?? "");
    return [control, () => {
      const entries = control.value.split(/[,，\s]+/).filter(Boolean);
      return entries.length ? entries : undefined;
    }];
  }
  control.value = value ?? "";
  if (kind === "textarea") return [control, () => control.value.trim() ? control.value : undefined];
  return [control, () => control.value.trim() || undefined];
}

function directionField(value) {
  const group = node("div", "model-choices");
  const checks = directionOptions.map(([direction, label]) => {
    const choice = node("label");
    const input = node("input");
    input.type = "checkbox";
    input.value = direction;
    input.checked = Array.isArray(value) && value.includes(direction);
    choice.append(input, document.createTextNode(label));
    group.append(choice);
    return input;
  });
  return [group, () => {
    const selected = checks.filter((checkbox) => checkbox.checked).map((checkbox) => checkbox.value);
    return selected.length ? selected : undefined;
  }];
}

function mapField(value) {
  const group = node("div", "model-map");
  const rows = new Map();
  function addRow(key = "", initial = "") {
    const row = node("div", "model-map-row");
    const keyInput = node("input");
    keyInput.placeholder = "参数名";
    keyInput.setAttribute("aria-label", "RapidOCR 参数名");
    keyInput.value = key;
    const kind = choices([["text", "文本"], ["number", "数字"], ["boolean", "开关"]]);
    kind.setAttribute("aria-label", "RapidOCR 参数类型");
    const valueCell = node("div");
    const complex = initial === null || typeof initial === "object";
    let read = () => initial;
    function render(kindName, startingValue) {
      const [input, getter] = scalar(kindName, startingValue);
      input.setAttribute("aria-label", "RapidOCR 参数值");
      if (kindName === "boolean") input.options[0].textContent = "请选择";
      valueCell.replaceChildren(input);
      read = kindName === "text" ? () => input.value : getter;
    }
    if (complex) {
      kind.add(new Option("复杂值", "complex"));
      kind.value = "complex";
      kind.disabled = true;
      valueCell.textContent = "保留已存值";
    } else {
      kind.value = ["number", "boolean"].includes(typeof initial) ? typeof initial : "text";
      render(kind.value, initial);
      kind.onchange = () => render(kind.value, undefined);
    }
    const remove = node("button", "small", "移除");
    remove.type = "button";
    remove.onclick = () => { rows.delete(row); row.remove(); };
    row.append(keyInput, kind, valueCell, remove);
    group.insertBefore(row, add);
    rows.set(row, () => [keyInput.value.trim(), read()]);
  }
  const add = node("button", "small", "添加参数");
  add.type = "button";
  add.onclick = () => addRow();
  group.append(add);
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const [key, item] of Object.entries(value)) addRow(key, item);
  }
  return [group, () => {
    const entries = new Map();
    for (const get of rows.values()) {
      const [key, item] = get();
      if (!key || item === undefined) throw new Error("请填写 RapidOCR 的参数名和对应值，或移除空行。");
      if (entries.has(key)) throw new Error("RapidOCR 参数名不能重复。");
      entries.set(key, item);
    }
    return entries.size ? Object.fromEntries(entries) : undefined;
  }];
}

export function parameterLabel(key, category) {
  if (key === "max_tokens" && Number(category) === 2) return "最大输入 Token 数";
  return fields[key]?.[0] || key;
}

export function renderModelParameterFields(container, option, existing = {}, isOci = false) {
  container.replaceChildren();
  const category = Number(option?.category), provider = option?.provider;
  const automaticEmbeddingMetadata = category === 2 && ["local_bge", "local_qwen"].includes(provider)
    || category === 3 && provider === "local_qwen";
  const readers = new Map();
  for (const key of option?.allowed_model_params || []) {
    if (automaticEmbeddingMetadata && ["model_revision", "processor_revision", "input_template"].includes(key)) continue;
    if (!Object.hasOwn(fields, key) || (!isOci && ociOnly.has(key))) continue;
    const [, kind, min] = fields[key];
    const group = node("div", "media-field");
    if (["textarea", "directions", "map"].includes(kind)) group.classList.add("full");
    const caption = node("label", "", parameterLabel(key, option.category));
    if (option.required_fields?.includes(`model_params.${key}`)) {
      const signedOnly = isOci && ociOnly.has(key) && option.secret_fields?.includes("api_key");
      caption.append(node("small", "model-field-help", signedOnly ? "（签名模式必填）" : "（必填）"));
    }
    const [control, read] = kind === "directions" ? directionField(existing[key])
      : kind === "map" ? mapField(existing[key]) : scalar(kind, existing[key], min);
    control.id = `model-param-${key}`;
    if (["directions", "map"].includes(kind)) {
      caption.id = `${control.id}-label`;
      control.setAttribute("role", "group");
      control.setAttribute("aria-labelledby", caption.id);
    } else {
      caption.htmlFor = control.id;
    }
    group.append(caption, control);
    if (key === "model_path" && automaticEmbeddingMetadata) {
      group.append(node("small", "model-field-help", "模型版本、预处理器版本和输入模板由模型服务根据本地文件及实际输入规则自动生成。"));
    }
    container.append(group);
    const initial = JSON.stringify(read());
    readers.set(key, () => {
      const result = read();
      return Object.hasOwn(existing, key) && JSON.stringify(result) === initial ? existing[key] : result;
    });
  }
  return {
    count: readers.size,
    values() {
      const params = { ...existing };
      for (const [key, read] of readers) {
        const value = read();
        if (value === undefined) delete params[key];
        else params[key] = value;
      }
      return params;
    },
  };
}
