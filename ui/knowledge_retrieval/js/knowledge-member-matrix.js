(() => {
"use strict";
// 两端共用同一套草稿语义：按人差量编辑，不改变 App 权限。
const memberOps = ['READ', 'UPLOAD', 'REVIEW', 'WITHDRAW', 'MANAGE']
const memberLabels = { READ: '读取', UPLOAD: '上传', REVIEW: '审核', WITHDRAW: '撤回', MANAGE: '管理' }
const batchOps = memberOps.slice(1, -1)
const pageSize = 10
const copy = grants => Object.assign(Object.create(null),
  Object.fromEntries(Object.entries(grants ?? {}).map(([id, ops]) => [id, ops.filter(op => memberOps.includes(op))])))

function createMemberMatrix(grants) {
  return {
    base: copy(grants), draft: copy(grants), options: null, optionById: new Map(), blocked: false, note: '',
    tab: 'members', query: '', page: 1, selections: { members: new Set(), candidates: new Set() }, batchOp: 'UPLOAD',
  }
}

function setOptions(s, options) {
  s.options = options
  s.optionById = new Map(options.map(person => [person.user_id, person]))
}
const member = (s, id) => s.optionById.get(id)
const present = (s, id) => Object.hasOwn(s.draft, id)
const selected = s => s.selections[s.tab]
const personName = (s, id) => member(s, id)?.display_name || '失效或未核验成员'

function matrixPage(s) {
  const query = s.query.trim().toLocaleLowerCase()
  const candidates = s.tab === 'members' ? Object.keys(s.draft) :
    (s.options ?? []).map(person => person.user_id).filter(id => !present(s, id))
  const ids = candidates
    .filter(id => id.toLocaleLowerCase().includes(query) || personName(s, id).toLocaleLowerCase().includes(query))
    .sort((a, b) => personName(s, a).localeCompare(personName(s, b), 'zh-CN') || a.localeCompare(b))
  const pages = Math.max(1, Math.ceil(ids.length / pageSize))
  const page = Math.min(s.page, pages)
  return { ids: ids.slice((page - 1) * pageSize, page * pageSize), count: ids.length, page, pages }
}

function changes(s) {
  return [...new Set([...Object.keys(s.base), ...Object.keys(s.draft)])].flatMap(id => {
    const before = s.base[id] ?? [], after = s.draft[id] ?? []
    const added = memberOps.filter(op => after.includes(op) && !before.includes(op))
    const removed = memberOps.filter(op => before.includes(op) && !after.includes(op))
    return added.length || removed.length || Object.hasOwn(s.base, id) !== present(s, id)
      ? [{ id, added, removed, removedMember: !present(s, id) }] : []
  })
}

const effectiveManagers = s => Object.keys(s.draft).filter(id => member(s, id)?.can_manage && s.draft[id].includes('MANAGE'))
const protectsManager = (s, id) => {
  const managers = effectiveManagers(s)
  return managers.length === 1 && managers[0] === id
}
const invalidGrant = (s, id) => !member(s, id) || (s.draft[id]?.includes('MANAGE') && !member(s, id)?.can_manage)

function memberWarning(s) {
  if (!s.options) return ''
  if (!effectiveManagers(s).length) return '至少保留一名具备 App 管理资格的库管理员。'
  if (Object.keys(s.draft).some(id => invalidGrant(s, id))) return '部分成员或管理资格已失效，请移除失效成员或取消其管理权限。'
  return ''
}

function adjust(s, id, op, on) {
  const person = member(s, id)
  if (!person || !present(s, id) || !memberOps.slice(1).includes(op) || s.blocked) return
  if (op === 'MANAGE' && (on ? !person.can_manage : protectsManager(s, id))) return
  const after = new Set(s.draft[id])
  if (on) after.add(op)
  else after.delete(op)
  s.draft[id] = memberOps.filter(value => after.has(value))
}

function join(s, ids) {
  if (s.blocked) return
  for (const id of ids) {
    if (!member(s, id) || present(s, id)) continue
    s.draft[id] = ['READ']
    s.selections.candidates.delete(id)
  }
  s.page = matrixPage(s).page
}

function remove(s, id) {
  if (!s.options || s.blocked || !present(s, id) || protectsManager(s, id)) return
  delete s.draft[id]
  s.selections.members.delete(id)
  s.page = matrixPage(s).page
}

function batch(s, on) {
  if (s.tab !== 'members' || !batchOps.includes(s.batchOp)) return
  for (const id of selected(s)) adjust(s, id, s.batchOp, on)
}

// 版本冲突后仅合并明确修改过的权限。若别人已经移除该成员，不自动将其加回。
function refreshMatrix(s, grants, options) {
  const next = copy(grants)
  const missing = []
  for (const change of changes(s)) {
    const { id, added, removed, removedMember } = change
    if (removedMember) { delete next[id]; continue }
    if (Object.hasOwn(s.base, id) && !Object.hasOwn(next, id)) { missing.push(id); continue }
    const ops = new Set(next[id] ?? ['READ'])
    for (const op of removed) ops.delete(op)
    for (const op of added) ops.add(op)
    next[id] = memberOps.filter(op => ops.has(op))
  }
  s.base = copy(grants)
  s.draft = next
  setOptions(s, options)
  s.blocked = false
  s.selections.members.clear()
  s.selections.candidates.clear()
  s.page = matrixPage(s).page
  s.note = '已读取最新授权并合并本次改动，请重新复核。'
  if (missing.length) s.note += ' 以下成员已被他人移除，未自动加回：' +
    missing.map(id => `${personName(s, id)}（${id}）`).join('、') + '。'
}

function reviewMembers(s) {
  const diff = changes(s)
  return {
    title: '确认授权变化',
    confirmLabel: '确认保存',
    scrollMessage: true,
    tone: diff.some(change => change.removedMember || change.removed.length) ? 'danger' : 'warning',
    message: [`本次影响 ${diff.length} 人。其他知识库和 App 角色不变。`, ...diff.map(({ id, added, removed, removedMember }) => {
      const description = removedMember ? '移除成员 · 撤销本库全部授权' :
        [added.length && '新增：' + added.map(op => memberLabels[op]).join('、'),
          removed.length && '撤销：' + removed.map(op => memberLabels[op]).join('、')].filter(Boolean).join('；')
      return `${personName(s, id)}（${id}）\n${description}`
    })].join('\n\n'),
  }
}

globalThis.KBotKnowledgeMemberMatrix = { memberOps, memberLabels, batchOps, createMemberMatrix, setOptions, member, present, selected, personName, matrixPage, changes,
  protectsManager, invalidGrant, memberWarning, adjust, join, remove, batch, refreshMatrix, reviewMembers }
})();
