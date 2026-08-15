import test from "node:test";
import assert from "node:assert/strict";
import {
  lockDetectedProfile,
  buildRollingContext,
  migrateRecognitionSettings,
} from "../src/recognition-profile-state.js";

test("auto locks once and manual profiles never change", () => {
  assert.equal(lockDetectedProfile("auto", null, "en-film"), "en-film");
  assert.equal(lockDetectedProfile("auto", "en-film", "fr-film"), "en-film");
  assert.equal(lockDetectedProfile("ja-film", null, "en-film"), null);
});

test("rolling context uses three latest original lines and 600 chars", () => {
  const subtitles = ["one", "two", "three", "x".repeat(700)].map((original) => ({ original }));
  const context = buildRollingContext(subtitles, 3, 600);
  assert.ok(!context.includes("one"));
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
