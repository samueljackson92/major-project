$(function () {
    $.getJSON("data.json", function(json) {

        var thumbs_directory = "/Volumes/Seagate/MammoData/thumbs/";

        $('#container').highcharts({
            chart: {
                type: 'scatter',
                zoomType: 'xy'
            },
            title: {
                text: 't-SNE'
            },
            xAxis: {
                title: {
                    enabled: true,
                    text: 'X'
                },
                startOnTick: true,
                endOnTick: true,
                showLastLabel: true
            },
            yAxis: {
                title: {
                    text: 'Y'
                }
            },
            legend: {
                layout: 'vertical',
                align: 'left',
                verticalAlign: 'top',
                x: 100,
                y: 70,
                floating: true,
                backgroundColor: (Highcharts.theme && Highcharts.theme.legendBackgroundColor) || '#FFFFFF',
                borderWidth: 1
            },
            plotOptions: {
                scatter: {
                    point: {
                        events: {
                            mouseOver: function () {
                                $("#image-viewer").html('<img src="' + thumbs_directory + this.name+'" style="display: block; margin-left: auto; margin-right: auto"/>')
                            }
                        }
                    },
                    marker: {
                        radius: 5,
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
    });
});
