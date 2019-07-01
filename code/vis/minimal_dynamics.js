/* Drawing 1D Dynamics
 *
 * Mostly a warmup for the more complex visualizations. At the moment, just
 * trying to get a static view of the dynamics in a 1D RNN with fixed dynamics.
 */

// Define background
var height = 300,
    width = 500;

var svg_elem = d3.select("body")
    .append("svg")
    .attrs({"width": width,
           "height": height});

svg_elem.selectAll("g")
    .data(["x_fun", "hx_scatter"]).enter()
    .append("g")
    .attr("id", (d) => d);

var scales = {
    "x_fun": {
        "time": d3.scaleLinear()
            .range([400, 500]),
        "value": d3.scaleLinear()
            .domain([-1, 1])
            .range([100, 0])
    },
    "hx": {
        "h": d3.scaleLinear()
            .domain([-1, 1])
            .range([0, 300]),
        "x": d3.scaleLinear()
            .domain([-1, 1])
            .range([200, 0])
    }
};

// Simulate some data, just for testing
var times = d3.range(0, 1, 0.05);
var N = times.length;

var dynamics = [];
for (var i = 0; i < N; i++) {
    dynamics.push({
        "time": times[i],
        "x": Math.sin(2 * Math.PI * times[i]),
        "h": times[i] ** 2
    });
}

// Display these data
var x_points = svg_elem.select("#x_fun")
    .selectAll("circles")
    .data(dynamics).enter()
    .append("circle")
    .attrs({
        "cx": (d) => scales.x_fun.time(d.time),
        "cy": (d) => scales.x_fun.value(d.x),
        "r": 3
    });

var time_pairs = [];
for (var i = 0; i < N - 1; i++) {
    time_pairs.push({
        "cur": dynamics[i],
        "next": dynamics[i + 1]
    });
}

var xh_segs = svg_elem.select("#hx_scatter")
    .selectAll("line")
    .data(time_pairs).enter()
    .append("line")
    .attrs({
        "x1": (d) => scales.hx.h(d["cur"]["h"]),
        "x2": (d) => scales.hx.h(d["next"]["h"]),
        "y1": (d) => scales.hx.x(d["cur"]["x"]),
        "y2": (d) => scales.hx.x(d["cur"]["x"]),
        "stroke": "#000",
        "stroke-width": 2
    });

var xh_starts = svg_elem.select("#hx_scatter")
    .selectAll("circle")
    .data(time_pairs).enter()
    .append("circle")
    .attrs({
        "cx": (d) => scales.hx.h(d["cur"]["h"]),
        "cy": (d) => scales.hx.x(d["cur"]["x"]),
        "r": 2
    });
