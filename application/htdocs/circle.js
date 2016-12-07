/**
 * Created by marcello on 7/12/16.
 */
function createFields(items) {
    console.log(items);
    $('.field').remove();
    var container = $('#container');
    var item_count = items.length;
    for(var i = 0; i < item_count; i++) {
        $('<div/>', {
            'class': 'field',
            'text': items[i].name,
        }).appendTo(container);
    }
}

function distributeFields() {
    var radius = 200;
    var fields = $('.field'), container = $('#container'),
        width = container.width(), height = container.height(),
        angle = 0, step = (2*Math.PI) / fields.length;
    fields.each(function() {
        var x = Math.round(width/2 + radius * Math.cos(angle) - $(this).width()/2);
        var y = Math.round(height/2 + radius * Math.sin(angle) - $(this).height()/2);
        if(window.console) {
            console.log($(this).text(), x, y);
        }
        $(this).css({
            left: x + 'px',
            top: y + 'px'
        });
        angle += step;
    });
}



