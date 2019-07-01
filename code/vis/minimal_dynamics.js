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
            .range([300, 0]),
        "x": d3.scaleLinear()
            .domain([-1, 1])
            .range([300, 0])
    }
};

// Simulate some data, just for testing
var times = d3.range(0, 1, 0.05);
var N = times.length;
var x = [];
for (var i = 0; i < N; i++) {
    x.push({
        "time": times[i],
        "value": Math.sin(2 * Math.PI * times[i])
    });
}

var h = [];
for (var i = 0; i < N; i++) {
    h.push({"value": d3.randomUniform()()});
}

// Display these data
var x_points = svg_elem.select("#x_fun")
    .selectAll("circles")
    .data(x).enter()
    .append("circle")
    .attrs({
        "cx": (d) => scales.x_fun.time(d.time),
        "cy": (d) => scales.x_fun.value(d.value),
        "r": 3
    });

var dynamics = [];
for (var i = 0; i < N; i++) {
    dynamics.push({
        "time": times[i],
        "x": x[i],
        "h": h[i]
    });
}

var time_pairs = [];
for (var i = 0; i < N - 1; i++) {
    time_pairs.push({
        "cur": dynamics[i],
        "next": dynamics[i + 1]
    });
}

var xh_segs = svg_elem.select("#hx_scatter")
    .selectAll("line")
    .data(time_pairs)
    .attrs({
        "x1": (d) => scales.hx.x(d["cur"]["x"]),
        "x2": (d) => scales.hx.x(d["cur"]["x"]),
        "y1": (d) => scales.hx.h(d["cur"]["h"]),
        "y2": (d) => scales.hx.h(d["next"]["h"]),
        "stroke": "#000"
    });

