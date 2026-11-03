// 日报编辑器只组织正式目录和版式，不在浏览器复制数据库判断规则。
export const editableRegions = layout => layout.sections.filter(item => !["OVERVIEW", "ACTIONS", "SQL", "BASELINE"].includes(item.section_id));
export const themes = catalog => catalog.report_sections.filter(item => !["OVERVIEW", "BASELINE", "ACTIONS"].includes(item.section_id));
export const checks = catalog => catalog.groups.flatMap(group => group.checks);
export const engines = catalog => [...new Set(checks(catalog).flatMap(check => check.supported_db_types))];
function policyIds(catalog) {
  const policy = catalog.daily_check_policy;
  if (!policy?.baseline || !policy?.supplemental || !policy?.required_metric_roles) throw new Error("检查目录未包含日报保障策略，请刷新目录后重试。");
  return new Set([...Object.values(policy.baseline), ...Object.values(policy.supplemental)].flat());
}
export function automaticIds(catalog) {
  const ids = policyIds(catalog);
  // 同主题同名检查沿用后端的工具和数据库范围合并口径，已保障的检查不再作为专项重复展示。
  const scope = check => JSON.stringify([check.report_section, check.display_name, check.tool_id, [...check.supported_db_types].map(value => value.toUpperCase()).sort()]);
  const covered = new Set(checks(catalog).filter(check => ids.has(check.check_id) && check.tool_id).map(scope));
  for (const check of checks(catalog)) {
    if (check.availability === "READY" && check.tool_id && covered.has(scope(check))) ids.add(check.check_id);
  }
  return ids;
}
export function checkGroups(catalog, topic, automatic) {
  const fixed = automaticIds(catalog), groups = new Map();
  for (const check of checks(catalog).filter(check => check.report_section === topic && fixed.has(check.check_id) === automatic)) {
    const key = check.display_name;
    if (!groups.has(key)) groups.set(key, {title:key, checks:[]});
    groups.get(key).checks.push(check);
  }
  return [...groups.values()];
}
export const groupIds = group => group.checks.filter(check => check.availability === "READY").map(check => check.check_id);
export const groupState = (group, selected) => {
  const ids = groupIds(group), count = ids.filter(id => selected.includes(id)).length;
  return {checked:ids.length > 0 && count === ids.length, mixed:count > 0 && count < ids.length, count, total:ids.length};
};
export function toggleGroup(group, selected, enabled) {
  const ids = groupIds(group);
  return enabled ? [...new Set([...selected, ...ids])] : selected.filter(id => !ids.includes(id));
}
export function selectedTopics(catalog, selectedIds) {
  const selected=new Set(selectedIds), known=checks(catalog).filter(check=>selected.has(check.check_id));
  const topics=themes(catalog).map(theme=>({title:theme.title,checks:[...new Set(known.filter(check=>check.report_section===theme.section_id).map(check=>check.display_name))]})).filter(theme=>theme.checks.length);
  const included=new Set(known.map(check=>check.check_id));
  const missing=selectedIds.filter(id=>!included.has(id));
  if(missing.length)topics.push({title:"未识别检查",checks:missing});
  return topics;
}
export function createDraft(catalog, template) {
  const automatic = policyIds(catalog);
  template = template?.draft || template?.definition || template;
  return {
    name: template?.display_name || "",
    selected: [...(template?.selected_check_ids || checks(catalog).filter(check => check.availability === "READY" && automatic.has(check.check_id)).map(check => check.check_id))],
    layout: JSON.parse(JSON.stringify(template?.daily_report_layout || catalog.daily_report_layout)),
  };
}
export function normalPreference(layout) {
  const modes = new Set(editableRegions(layout).map(item => item.normal_detail));
  return modes.size === 1 ? [...modes][0] : "CUSTOM";
}
export function setNormalPreference(layout, mode) {
  if (mode !== "CUSTOM") editableRegions(layout).forEach(item => {item.normal_detail = mode;});
}
export function changeSummary(draft, original) {
  const added = draft.selected.filter(id => !original.selected.includes(id)).length;
  const removed = original.selected.filter(id => !draft.selected.includes(id)).length;
  return {added, removed, layoutChanged:JSON.stringify(draft.layout) !== JSON.stringify(original.layout)};
}
export function unavailableIds(catalog, selected) {
  const ready = new Set(checks(catalog).filter(check => check.availability === "READY").map(check => check.check_id));
  return selected.filter(id => !ready.has(id));
}
export function canConfigureStatistics(catalog, regionId) {
  return Boolean(catalog.report_sections.find(item => item.section_id === regionId)?.semantic_roles?.length);
}
export const fixedDescriptions = {
  OVERVIEW: ["整体结论与检查边界", "重要问题与专题定位", "必要检查完成情况"],
  ACTIONS: ["按优先级汇总处置事项", "关联具体问题与影响对象", "同一判断依据下的复查要求"],
  SQL: ["依据可取得的耗时、CPU、I/O、锁相关等待、扫描量和临时空间等指标筛查", "仅列已识别的问题 SQL，按严重性排序，不因执行次数多而入选", "逐条保留问题原因、Agent 初步分析与优化建议", "检查完整且无问题时显示“当前没有问题 SQL”"],
};
