<!DOCTYPE html>
<html lang="en">
<head>
        <?php 
    require 'csfs_strings.php'; 
    ?>
    <meta charset="UTF-8">
    <title><?php echo $S['TITLE']; ?></title>
    <script src="http://rubaxa.github.io/Sortable/Sortable.js"></script>
    <script src="https://code.jquery.com/jquery-3.1.1.min.js"
            integrity="sha256-hVVnYaiADRTO2PzUGmuLJr8BLUSjGIZsDYGmIJLv2b8="
            crossorigin="anonymous"></script>
    <!-- Latest compiled and minified CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"
          integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">

    <!-- Optional theme -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css"
          integrity="sha384-rHyoN1iRsVXV4nD0JutlnGaslCJuC7uwjduW9SVrLvRYooPp2bWYgmgJQIXwl/Sp" crossorigin="anonymous">

    <!-- Latest compiled and minified JavaScript -->
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"
            integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa"
            crossorigin="anonymous"></script>
    <script src="circle.js"></script>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css"/>
    <link rel="stylesheet" href="style.css"/>


</head>
<body>
    
    <?php 
    require 'database.php'; 
    
    $data = get_data('student', 'layperson');
    $task = get_task('student', 'layperson');
    ?>
    


<div class="container-fluid">
    <h1><?php echo $S['HEADER']; ?></h1>
    
    <div class="col-md-12">
        <h2><?php echo $S['TASK_DESCRIPTION'] ?></h2>
        <p> <?php echo $task['description']; ?> </p>
    </div>
    <div class="col-md-4 col-xs-6">
        <div class="box" id="target">
        </div>
    </div>
    <div class="col-md-8 col-xs-6">
        <div class="box" id="source">
        </div>
    </div>
    <div class="col-md-12">
        <input id="output" class="form-control" type="text" readonly="readonly">
    </div>
    <div class="col-md-12">
        <button id="reset" type="button" class="btn">Reset</button>
        <button id="submit" type="button" class="btn btn-primary" disabled="disabled">Finished</button>
    </div>
</div>


<script>
    var items = <?php echo json_encode($data); ?>;
    var CSFS = {'items': items};

    function prepend_numbering(){
        var $rows = $('#target').find('.row');
        for (var i = 0; i < $rows.length; i++){
            if ($rows.eq(i).find('.number').length == 0){
                $rows.eq(i).find('.feature').before($('<span class="number"></span>'));
            }
            console.log($rows.eq(i).find('span').eq(2).text());
            $rows.eq(i).find('.number').text((i+1)+'. ');
        }
        console.log('---');
    }

    function onAdded(){
        $('#target').find('.item').attr('class', 'row item');

        prepend_numbering();
            if (is_count_correct(CSFS.items)) {
                $('#submit').attr('disabled', false);
            }
    }
    function  onFinished() {
        if (is_count_correct(CSFS.items)) { // is valid
            
            var list_ordered = get_ordered();
            var output = get_output_string(list_ordered);
            $('#output').val(output);
            return true;
        }
    }

    function create_item(item) {
        return $('<div role="button" class="col-md-3 item"><span  class="glyphicon glyphicon-move handle" aria-hidden="true"></span> <span class="feature" id="' + item.No + '">' + item.Name + '</span> <span class="glyphicon glyphicon-question-sign" data-toggle="tooltip" title="' + item.Description + '"></span></div>');
    }

    function clear_all() {
        $('#output').val('');
        $('#source').empty();
        $('#target').empty();
        $('#submit').attr('disabled', 'disabled');
    }

    function reset() {
        clear_all();

        for (var i = 0; i < CSFS.items.length; i++) {
            var $item = create_item(CSFS.items[i]);
            $('#source').append($item);
        }

        $('#source .item').dblclick(function(){
            $('#target').append($(this));
            onAdded();
        });

        var source = document.getElementById('source');
        var target = document.getElementById('target');
        Sortable.create(
                source,
                {
                    group: "ranking-source",
                    draggable: '.item',
                    handle: '.handle',
                    animation: 200,
                }
        );
        Sortable.create(target,
                {
                    group: {name: "ranking-target", put: ["ranking-source"]},
                    draggable: '.item',
                    ghostclass: 'ghost',
                    animation: 300,
                    onAdd: onAdded,
                    onMove: onAdded,
                });
                
    }

    (function () {
        reset();

        $('#submit').click(function (e) {
            onFinished();
        });

        $('#reset').click(function () {
            reset();
        });
        
                
        $('[data-toggle="tooltip"]').tooltip(); 
    })();

    function get_output_string(list_ordered) {
        var output = "";
        for (var i = 0; i < list_ordered.length; i++) {
            output += '"';
            output += i;
            output += ":";
            output += list_ordered[i].id;
            output += '"';
            if (i<list_ordered.length-1) {
                output += ",";
            }
        }
        output += "|";
        output += "MD5(...)";
        return output;
    }

    function is_count_correct(items) {
        var list_ordered = get_ordered();
        return list_ordered.length === items.length;
    }

    function get_ordered() {
        var $target = $('#target');
        var features = $target.find('span.feature');
        var list = [];
        for (var i = 0; i < features.length; i++) {
            var id = features[i].id;
            var name = features[i].innerHTML;
            list.push({'id': id, 'name': name});
        }
        return list;
    }
</script>
</body>
</html>