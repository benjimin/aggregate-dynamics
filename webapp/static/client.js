function plot(container) {
    // create image
    var svg = container.append("svg")
                                    .attr("width", container.width())
                                    .attr("height", container.height());

    let m = {"left": 100, "bottom": 20, "right": 20, "top": 20},
        w = container.width() - m.left - m.right,
        h = container.height() - m.top - m.bottom;

    // scaling functions
    var x = d3.scaleTime().range([0, w]),
        z = x.copy(),
        y = d3.scaleLinear().range([h, 0]);

    // groups for axis marks and canvas
    function tr(tx, ty) { return "translate(" + [tx, ty].toString() + ")" };
    let plotregion = svg.append("g").attr("transform", tr(m.left, m.top)),
        bottom = svg.append("g").attr("transform", tr(m.left, h + m.top)),
        side = svg.append("g").attr("transform", tr(m.left, m.top));

    // label of vertical axis
    plotregion.append('g').attr("transform", tr(-50, h/2))
        .append("text").attr("text-anchor", "middle")
        .attr("transform","rotate(-90)").text("water (hectares)")

    // define clipping area (for reference by plot objects)
    plotregion.append("defs").append("clipPath").attr("id", "cliparea")
            .append("rect").attr("width", w).attr("height", h);

    // border around canvas
    var r = plotregion.append("rect").attr("width", w).attr("height", h)
        .style("fill-opacity", 0).style("stroke", "black");

    // functions for generating paths from data, using scales
    var estimate = d3.line()
                         .x(d => x(d[0]))
                         .y(d => y(d[2]));
    var uncertainty = d3.area()
                         .x(d => x(d[0]))
                         .y0(d => y(d[1]))
                         .y1(d => y(d[3]));

    // path depictions
    var curve = plotregion.append("path").style("fill", "none")
                    .style("stroke", "midnightblue")
                    .style("stroke-width", 2)
                    .style("clip-path", "url(#cliparea)");
    var envelope = plotregion.append("path")
                    .style("fill", "steelblue").style("stroke", "none")
                    .style("fill-opacity", 0.5)
                    .style("clip-path", "url(#cliparea)");


    let zoomcatcher = d3.zoom().on("zoom", zoomhandler)
            .scaleExtent([1, Infinity]).translateExtent([[0, 0], [w, h]])
            .extent([[0, 0], [w, h]]);

    r.call(zoomcatcher);

    this.update = function(data) {
        curve.datum(data);
        envelope.datum(data);
        x.domain([data[0][0], data.slice(-1)[0][0]]);
        z = x.copy;
        redraw();
    };

    function redraw() {
        curve.attr("d", estimate);
        envelope.attr("d", uncertainty);
        bottom.call(xaxismaker);
        side.call(yaxismaker);
    };

    function zoomhandler() {
        x.domain((d3.event).transform.rescaleX(z).domain());
        redraw();
    };


};