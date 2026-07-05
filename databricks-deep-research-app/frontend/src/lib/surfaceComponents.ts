/**
 * Single source of truth for surface component *categories* used by render-time
 * layout inference. Keep in sync with the backend catalog
 * (src/deep_research/surface/catalog.py): INPUT_COMPONENTS mirrors it exactly;
 * RESULT_COMPONENTS is a render-only grouping (see note) with no 1:1 backend twin.
 */

/** Two-way input controls (mirror surface/catalog.py INPUT_COMPONENTS). */
export const INPUT_COMPONENTS: ReadonlySet<string> = new Set([
  'TextField',
  'TextArea',
  'Select',
  'Checkbox',
]);

/**
 * Components that mark a subtree as "results" for host section inference.
 * Intentionally broader than the backend OUTPUT_COMPONENTS set: it also includes
 * StatusBadge / ReportRegion / Tabs / TabPane, which only ever appear in the
 * results area. Divergence is deliberate — this drives the Inputs-vs-Results
 * split, not structured-output slot collection.
 */
export const RESULT_COMPONENTS: ReadonlySet<string> = new Set([
  'ReportRegion',
  'StatusBadge',
  'Table',
  'MetricGrid',
  'KeyFindings',
  'Chart',
  'List',
  'Tabs',
  'TabPane',
]);
