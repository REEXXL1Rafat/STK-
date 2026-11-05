// SATARK Risk Window: which fields are already cut, from Sentinel-2.
//
// How to run (about 5 minutes of clicking, then Google does the work, often 1 to 3 hours):
//   1. Open https://code.earthengine.google.com and paste this whole file into a new script.
//   2. Press Run. Two exports appear in the Tasks tab on the right.
//   3. Press RUN on each one, then OK. They save CSV files into a "SATARK" folder in your Google Drive.
//   4. When both say Completed, tell Claude. It can read them from your Drive.
//
// What it measures: for every 5 km area and every 5 days of the burning season, the share of
// paddy that was green at its peak and is now cut (low greenness). Cut but not yet cleared paddy is
// what gets burned, so this tells the forecast where fields are ready to burn.

var CELL = 0.05;          // same 5 km grid as the forecast
var STEP = 5;             // a new reading every 5 days
var LOOKBACK = 10;        // each reading uses the clear images from the last 10 days
var YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026];

var REGIONS = {
  punjab_haryana: {
    bbox: [73.8, 27.6, 77.6, 32.6], states: ['Punjab', 'Haryana'],
    peak: ['08-20', '09-25'], start: '09-15', end: '12-10', nextYearEnd: false
  },
  purba_medinipur: {
    bbox: [86.9, 21.5, 88.3, 22.7], states: ['West Bengal'],
    peak: ['09-20', '10-30'], start: '10-25', end: '01-31', nextYearEnd: true
  }
};

var crop = ee.ImageCollection('ESA/WorldCover/v200').first().eq(40);        // cropland
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');  // cloud score

function ndvi(geom, start, end) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geom)
    .filterDate(start, end)
    .linkCollection(csPlus, ['cs_cdf'])
    .map(function (img) {
      return img.normalizedDifference(['B8', 'B4']).rename('ndvi')
        .updateMask(img.select('cs_cdf').gte(0.6));                          // keep clear pixels only
    });
}

function cellsFor(cfg) {
  var w = cfg.bbox[0], s = cfg.bbox[1];
  var ny = Math.ceil((cfg.bbox[3] - s) / CELL), nx = Math.ceil((cfg.bbox[2] - w) / CELL);
  var states = ee.FeatureCollection('FAO/GAUL/2015/level1')
    .filter(ee.Filter.eq('ADM0_NAME', 'India'))
    .filter(ee.Filter.inList('ADM1_NAME', cfg.states));
  var list = ee.List.sequence(0, ny - 1).map(function (iy) {
    return ee.List.sequence(0, nx - 1).map(function (ix) {
      var x0 = ee.Number(ix).multiply(CELL).add(w);
      var y0 = ee.Number(iy).multiply(CELL).add(s);
      return ee.Feature(ee.Geometry.Rectangle([x0, y0, x0.add(CELL), y0.add(CELL)], null, false),
                        {iy: ee.Number(iy).int(), ix: ee.Number(ix).int()});
    });
  }).flatten();
  return ee.FeatureCollection(list).filterBounds(states.geometry());
}

function seasonTable(name, cfg, year, cells) {
  var geom = cells.geometry().bounds();
  var peak = ndvi(geom, year + '-' + cfg.peak[0], year + '-' + cfg.peak[1]).max();
  var paddy = crop.and(peak.gt(0.55));                                        // was green this season
  var start = ee.Date(year + '-' + cfg.start);
  var end = ee.Date((cfg.nextYearEnd ? year + 1 : year) + '-' + cfg.end);
  var n = end.difference(start, 'day').divide(STEP).floor();
  var steps = ee.List.sequence(0, n);
  return ee.FeatureCollection(steps.map(function (k) {
    var d = start.advance(ee.Number(k).multiply(STEP), 'day');
    var now = ndvi(geom, d.advance(-LOOKBACK, 'day'), d).median();
    var cut = now.lt(0.35).rename('cut').updateMask(paddy).updateMask(now.mask());
    var stats = cut.reduceRegions({
      collection: cells, reducer: ee.Reducer.mean().combine(ee.Reducer.count(), '', true),
      scale: 60, tileScale: 4
    });
    return stats.map(function (f) { return f.set('date', d.format('YYYY-MM-dd')).setGeometry(null); });
  })).flatten();
}

Object.keys(REGIONS).forEach(function (name) {
  var cfg = REGIONS[name];
  var cells = cellsFor(cfg);
  var all = ee.FeatureCollection(YEARS.map(function (y) { return seasonTable(name, cfg, y, cells); })).flatten();
  Export.table.toDrive({
    collection: all,
    description: 'satark_harvest_' + name,
    folder: 'SATARK',
    fileNamePrefix: 'harvest_' + name,
    fileFormat: 'CSV',
    selectors: ['iy', 'ix', 'date', 'mean', 'count']
  });
  print(name + ': ' + YEARS.length + ' seasons queued. Press RUN in the Tasks tab.');
});
