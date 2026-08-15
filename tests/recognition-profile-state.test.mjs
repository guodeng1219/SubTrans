import test from "node:test";
import assert from "node:assert/strict";
import {
  lockDetectedProfile,
  buildRollingContext,
  migrateRecognitionSettings,
  resetRecognitionSession,
  applyDetectedProfileForSession,
  normalizeRecognitionSettingsForCatalog,
  resolveRecognitionSettingsAfterCatalog,
} from "../src/recognition-profile-state.js";

test("auto locks once and manual profiles never change", () => {
  assert.equal(lockDetectedProfile("auto", null, "en-film"), "en-film");
  assert.equal(lockDetectedProfile("auto", "en-film", "fr-film"), "en-film");
  assert.equal(lockDetectedProfile("ja-film", null, "en-film"), null);
});

test("rolling context uses three latest original lines and 600 chars", () => {
  const subtitles = [
    { end: 1, original: "one" },
    { end: 2, original: "two" },
    { end: 3, original: "three" },
    { end: 4, original: "x".repeat(700) },
    { end: 101, original: "future dialogue" },
  ];
  const context = buildRollingContext(subtitles, 10, 3, 600);
  assert.ok(!context.includes("one"));
  assert.ok(!context.includes("future dialogue"));
  assert.ok([...context].length <= 600);
});

test("version one settings migrate without losing uncommon languages", () => {
  assert.deepEqual(migrateRecognitionSettings({ sourceLang: "en" }), {
    recognitionProfileId: "en-film",
    accentVariant: "auto",
    sourceLang: "en",
  });
  assert.equal(migrateRecognitionSettings({ sourceLang: "es" }).recognitionProfileId, "custom");
  assert.equal(migrateRecognitionSettings({ sourceLang: "" }).recognitionProfileId, "auto");
});

test("new recognition session clears only the automatic lock", () => {
  assert.deepEqual(resetRecognitionSession("en-film", "en-gb"), {
    selectedProfileId: "en-film",
    lockedProfileId: null,
    accentVariant: "en-gb",
  });
});

test("stale chunk responses cannot change the automatic lock", () => {
  assert.equal(applyDetectedProfileForSession(2, 1, "auto", null, "en-film"), null);
  assert.equal(applyDetectedProfileForSession(2, 2, "auto", null, "en-film"), "en-film");
  assert.equal(applyDetectedProfileForSession(2, 2, "auto", "en-film", "fr-film"), "en-film");
});

test("project profile waits for catalog and degrades without losing source language", () => {
  const migrated = migrateRecognitionSettings({ sourceLang: "en" });
  assert.equal(
    normalizeRecognitionSettingsForCatalog(migrated, new Set(["auto", "custom", "en-film"]))
      .recognitionProfileId,
    "en-film"
  );
  const fallback = normalizeRecognitionSettingsForCatalog(
    migrated,
    new Set(["auto", "custom"])
  );
  assert.equal(fallback.recognitionProfileId, "custom");
  assert.equal(fallback.sourceLang, "en");
  assert.equal(fallback.accentVariant, "auto");
});

test("project recognition settings are not resolved before the catalog", async () => {
  let releaseCatalog;
  const catalogReady = new Promise((resolve) => {
    releaseCatalog = resolve;
  });
  let settled = false;
  const pending = resolveRecognitionSettingsAfterCatalog(
    { sourceLang: "en" },
    catalogReady
  ).then((value) => {
    settled = true;
    return value;
  });
  await Promise.resolve();
  assert.equal(settled, false);
  releaseCatalog(new Set(["auto", "custom", "en-film"]));
  assert.equal((await pending).recognitionProfileId, "en-film");
});

test("rolling context excludes untimed and future dialogue", () => {
  const context = buildRollingContext(
    [{ original: "untimed" }, { end: 9.5, original: "before" }, { end: 10.5, original: "after" }],
    10,
    3,
    600
  );
  assert.ok(!context.includes("untimed"));
  assert.ok(context.includes("before"));
  assert.ok(!context.includes("after"));
});
