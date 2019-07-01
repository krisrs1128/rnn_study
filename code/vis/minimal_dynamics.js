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

var scales = {
    "x_fun": {
        "time": d3.scaleLinear()
            .range([400, 500]),
        "value": d3.scaleLinear()
            .domain([-1, 1])
            .range([100, 0])
    }
};

// Define function values
var times = d3.range(0, 1, 0.05);
var x = [];
for (var i = 0; i < times.length; i++) {
    x.push({
        "time": times[i],
        "value": Math.sin(2 * Math.PI * times[i])
    });
}

var x_points = svg_elem.selectAll("circles")
    .data(x).enter()
    .append("circle")
    .attrs({
        "cx": (d) => scales.x_fun.time(d.time),
        "cy": (d) => scales.x_fun.value(d.value),
        "r": 3
    });
