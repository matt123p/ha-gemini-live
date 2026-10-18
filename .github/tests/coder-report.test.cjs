const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { test } = require('node:test');

// Execute the actual inline publication step, including its prompt/run checks,
// against fake APIs. No network, Coder credentials, or issue writes are needed.
const workflow = readFileSync(join(__dirname, '../workflows/coder.yml'), 'utf8')
  .replace(/\r\n/g, '\n');
const header = '\n          script: |\n';
const script = workflow.slice(workflow.lastIndexOf(header) + header.length)
  .split('\n').map(line => line.replace(/^ {12}/, '')).join('\n');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const publish = new AsyncFunction('github', 'context', 'process', 'fetch', script);
const reference = '39030271457-1';
const start = `<!-- triage-report:${reference}:start -->`;
const end = `<!-- triage-report:${reference}:end -->`;
const report = '## Summary\n\nIssue #13 describes a failed voice turn.\n\n## Findings\n\nEvidence and hypotheses.';
const block = `${start}\n${report}\n${end}`;
const message = (id, role, value) => ({ id, role, content: [{ type: 'text', text: value }] });

async function run(reply, { promptReference = reference, previous = [] } = {}) {
  const writes = [];
  const messages = [
    ...previous,
    message(10, 'user', `Investigate issue.\nPublication reference: ${promptReference}`),
    message(11, 'assistant', reply),
  ];
  const github = {
    rest: { issues: {
      listComments() {},
      async createComment(value) { writes.push(value); },
      async updateComment(value) { writes.push(value); },
    } },
    async paginate() { return []; },
  };
  let error;
  try {
    await publish(github, {
      runId: 39030271457, repo: { owner: 'matt123p', repo: 'ha-gemini-live' },
      issue: { number: 13 },
    }, { env: {
      CODER_URL: 'https://coder.example.test', CODER_TOKEN: 'test-token',
      CHAT_ID: '11111111-2222-3333-4444-555555555555',
      PUBLICATION_REFERENCE: reference,
    } }, async () => ({ ok: true, json: async () => ({ messages, has_more: false }) }));
  } catch (value) {
    error = value;
  }
  return { writes, error };
}

test('publishes the reported response with a saved-file preamble and a full report', async () => {
  const { writes, error } = await run(
    'The report is saved at `/home/coder/triage/issues/matt123p/ha-gemini-live/13/PLAN.md`. Full text follows.\n\n' + block
  );
  assert.ifError(error);
  assert.equal(writes.length, 1);
  assert.ok(writes[0].body.endsWith(report));
  assert.ok(!writes[0].body.includes('/home/coder'));
  assert.ok(!writes[0].body.includes('<!-- triage-report:'));
});

test('accepts a bare report and Windows line endings with surrounding whitespace', async () => {
  for (const reply of [block, ('Intro\n  ' + block + '  \nFinished.').replace(/\n/g, '\r\n')]) {
    const { writes, error } = await run(reply);
    assert.ifError(error);
    assert.equal(writes.length, 1);
    assert.ok(writes[0].body.endsWith(report));
    assert.ok(!writes[0].body.includes('Finished.'));
  }
});

for (const [name, reply] of Object.entries({
  'clarification without a report': 'Which version should I investigate?',
  'missing end marker': `${start}\n${report}`,
  'missing start marker': `${report}\n${end}`,
  'reversed markers': `${end}\n${report}\n${start}`,
  'empty report': `${start}\n   \n${end}`,
  'duplicate start marker': `${start}\n${block}`,
  'duplicate end marker': `${block}\n${end}`,
  'multiple reports': `${block}\n${block}`,
  'old run attempt': block.replaceAll(reference, '39030271457-0'),
  'markers embedded in prose': `Intro ${start}\n${report}\n${end}`,
})) {
  test(`refuses ${name} without posting a comment`, async () => {
    const { writes, error } = await run(reply);
    assert.match(error?.message || '', /No complete report/);
    assert.equal(writes.length, 0);
  });
}

test('does not publish a valid block against a different run prompt', async () => {
  const { writes, error } = await run(block, { promptReference: 'another-run' });
  assert.match(error?.message || '', /Current triage prompt was not found/);
  assert.equal(writes.length, 0);
});

test('does not fall back to an earlier report when the final reply asks a question', async () => {
  const { writes, error } = await run('Please clarify the version.', {
    previous: [message(1, 'assistant', block)],
  });
  assert.match(error?.message || '', /No complete report/);
  assert.equal(writes.length, 0);
});
