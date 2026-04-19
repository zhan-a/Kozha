/**
 * Body-region polygon map for the CWASA/JASigning avatar preview.
 *
 * The CWASA avatar renders into a <canvas>, and Kozha's preview panel
 * may also show a pre-rendered <video> for the same sign. Raycasting
 * into the third-party CWASA scene graph is not feasible (we don't
 * own the bone rig), and the two render modes share the same standing
 * pose, so both cases use the same 2-D SVG polygon overlay.
 *
 * All polygons live in a normalized SVG viewBox of 100×120 (5:6
 * portrait). The overlay is stretched via ``preserveAspectRatio="none"``
 * over whatever preview element is visible, so a single calibration
 * works across aspect ratios. Coordinates assume:
 *
 *   - The avatar is front-facing, feet roughly at y=120.
 *   - The DOMINANT hand hangs on the viewer's LEFT (avatar's right),
 *     which matches the default CWASA pose for right-handed avatars.
 *   - "Neutral space" is the air in front of the trunk where many
 *     signs are articulated when no body contact is needed.
 *
 * Neighboring polygons may overlap slightly — point-in-polygon hit
 * testing is first-match, and the order below is meaningful: more
 * specific hands/forearms win over the trunk they overlap with.
 * Keep that ordering in mind if you add or reorder regions.
 *
 * Calibrate visually via ``?debug=1`` on the authoring page — the
 * overlay is drawn in a translucent fill so polygons can be matched
 * against the live avatar image.
 */
(() => {
  'use strict';

  const VIEWBOX = { width: 100, height: 120 };

  /**
   * Region definitions, in hit-testing order (first match wins).
   *
   * Each entry:
   *   - id: machine name sent as ``target_region`` to /correct.
   *   - label: human-facing label for the dropdown + aria-label.
   *   - points: array of [x, y] in the 100×120 viewBox.
   *   - z: draw-order hint (higher = above). Only used for visual
   *        calibration; hit testing uses array order.
   */
  const REGIONS = [
    // --- Hands and forearms: evaluated first so they win over the
    //     trunk / neutral-space regions they sit on top of. ---
    {
      id: 'hand-dom',
      label: 'Dominant hand',
      points: [[14, 78], [30, 78], [32, 94], [16, 94]],
      z: 5,
    },
    {
      id: 'hand-nondom',
      label: 'Non-dominant hand',
      points: [[68, 78], [84, 78], [86, 94], [70, 94]],
      z: 5,
    },
    {
      id: 'forearm-dom',
      label: 'Dominant forearm',
      points: [[18, 58], [32, 58], [30, 78], [14, 78]],
      z: 4,
    },
    {
      id: 'forearm-nondom',
      label: 'Non-dominant forearm',
      points: [[66, 58], [80, 58], [84, 78], [68, 78]],
      z: 4,
    },
    {
      id: 'upper-arm-dom',
      label: 'Dominant upper arm',
      points: [[22, 40], [36, 40], [32, 58], [18, 58]],
      z: 4,
    },
    {
      id: 'upper-arm-nondom',
      label: 'Non-dominant upper arm',
      points: [[62, 40], [76, 40], [80, 58], [66, 58]],
      z: 4,
    },
    // --- Head: split into face-upper and face-lower for finer targeting,
    //     with a residual 'head' polygon for the crown/back. ---
    {
      id: 'face-upper',
      label: 'Face (upper)',
      points: [[36, 10], [62, 10], [64, 20], [34, 20]],
      z: 3,
    },
    {
      id: 'face-lower',
      label: 'Face (lower)',
      points: [[34, 20], [64, 20], [62, 30], [36, 30]],
      z: 3,
    },
    {
      id: 'head',
      label: 'Head',
      points: [[34, 2], [64, 2], [64, 10], [36, 10], [36, 30], [34, 30]],
      z: 2,
    },
    // --- Neck and trunk. ---
    {
      id: 'neck',
      label: 'Neck',
      points: [[42, 30], [56, 30], [56, 38], [42, 38]],
      z: 3,
    },
    {
      id: 'trunk',
      label: 'Trunk / chest',
      points: [[36, 38], [62, 38], [62, 76], [36, 76]],
      z: 2,
    },
    // --- Neutral space: the air in front of the signer. Three
    //     horizontal bands (low, mid, high) matching typical HamNoSys
    //     location conventions for signs articulated away from the body. ---
    {
      id: 'neutral-space-high',
      label: 'Neutral space (high)',
      points: [[20, 30], [80, 30], [80, 46], [20, 46]],
      z: 1,
    },
    {
      id: 'neutral-space-mid',
      label: 'Neutral space (mid)',
      points: [[20, 46], [80, 46], [80, 64], [20, 64]],
      z: 1,
    },
    {
      id: 'neutral-space-low',
      label: 'Neutral space (low)',
      points: [[20, 64], [80, 64], [80, 80], [20, 80]],
      z: 1,
    },
  ];

  // Build a lookup keyed by id for fast access.
  const REGION_BY_ID = Object.fromEntries(REGIONS.map((r) => [r.id, r]));

  /**
   * Ray-casting point-in-polygon, following the wind-count algorithm.
   * The polygon is an array of [x, y] vertices; the edges are implicit
   * (last vertex wraps to the first).
   */
  function pointInPolygon(x, y, points) {
    let inside = false;
    const n = points.length;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const [xi, yi] = points[i];
      const [xj, yj] = points[j];
      const intersects =
        (yi > y) !== (yj > y) &&
        x < ((xj - xi) * (y - yi)) / (yj - yi || 1e-9) + xi;
      if (intersects) inside = !inside;
    }
    return inside;
  }

  /**
   * Hit-test a point in viewBox coordinates against every region.
   * Returns the first (order-sensitive) match's ID, or null.
   */
  function hitTest(x, y) {
    for (const region of REGIONS) {
      if (pointInPolygon(x, y, region.points)) return region.id;
    }
    return null;
  }

  /**
   * Convert a polygon to an SVG ``points="x1,y1 x2,y2…"`` attribute string.
   */
  function polygonAttr(points) {
    return points.map((p) => p.join(',')).join(' ');
  }

  const API = {
    VIEWBOX,
    REGIONS,
    REGION_BY_ID,
    hitTest,
    pointInPolygon,
    polygonAttr,
  };

  // UMD-style export: attach to window and to module.exports if present
  // (so the Playwright test can eval region metadata via page.evaluate).
  if (typeof window !== 'undefined') window.C2H_REGIONS = API;
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
})();
