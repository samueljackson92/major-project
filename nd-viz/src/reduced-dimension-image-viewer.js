var config = {};

function visualise(cfg)
{
    config = cfg;
    $.getJSON(config.mapping, plotMapping);
}

function plotMapping(json) {

    $('#container').highcharts({
        chart: {
            type: 'scatter',
            zoomType: 'xy'
        },
        title: {
            text: config.title
        },
        xAxis: {
            title: {
                text: config.xAxisName
            },
            startOnTick: true,
            endOnTick: true,
            showLastLabel: true,
        },
        yAxis: {
            title: {
                text: config.yAxisName
            },
            gridLineWidth: 0,
        },
        legend: {
            enabled: config.showLegend,
            layout: 'vertical',
            align: 'left',
            verticalAlign: 'top',
            x: 100,
            y: 100,
            floating: true,
            backgroundColor: (Highcharts.theme &&
                              Highcharts.theme.legendBackgroundColor) ||
                              '#FFFFFF',
            borderWidth: 1
        },
        plotOptions: {
            scatter: {
                point: {
                    events: {
                        mouseOver: onDataPointHover
                    }
                },
                marker: {
                    radius: 3,
                    states: {
                        hover: {
                            enabled: true,
                            lineColor: 'rgb(100,100,100)'
                        }
                    }
                },
                states: {
                    hover: {
                        marker: {
                            enabled: false
                        }
                    }
                },
            }
        },
        series: json
    });
}

function onDataPointHover() {
    var html = '<img src="' + config.imageDirectory + this.name +
               '" style="display: block; margin-left: auto; margin-right: auto"/>'
    $("#image-viewer").html(html);
}
