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
    <link rel="stylesheet" type="text/css" href="style.css"/>


</head>
<body>
    
    <?php 
    require 'database.php'; 
    $dataset_name = 'test';
    $condition = 'layperson';
    
    
    $data = get_data($dataset_name, $condition);
    $task = get_task('student', $condition);
    
    $name = filter_input(INPUT_GET, 'name', FILTER_SANITIZE_STRING);
    $token = filter_input(INPUT_GET, 'token', FILTER_SANITIZE_STRING);
    ?>
    
<?php if ($_SERVER['REQUEST_METHOD'] === 'POST') { 
    $name = filter_input(INPUT_POST, 'name', FILTER_SANITIZE_STRING);
     $result_token = handle_post($dataset_name, $condition);
     $token = filter_input(INPUT_POST, 'token', FILTER_SANITIZE_STRING);
     
    ?>
    <h1><?php echo $S['THANKS']; ?>, <?php echo $name; ?>!</h1>
    <p><?php echo $S['TASK_AFTER_SUBMIT']; ?></p>
    
    <div class="col-md-3 col-xs-12">
        Token: 
    </div>
    <div class="col-md-9 col-xs-12">
        <input class="form-control" type="text" name="output_token" value="<?php echo $result_token; ?>" readonly="readonly">
    </div>
    <?php
    
} else {
    // Is not post
    ?>    
<div class="container-fluid">
    <h1><?php echo $S['HEADER']; ?></h1>
    
    <div class="col-md-12">
        <h2><?php echo $S['TASK_DESCRIPTION'] ?></h2>
        <p> <?php echo $task['description']; ?> </p>
    </div>
    <form method="post" id="form" action="">
        
        <div class="col-md-12"><div class="alert alert-danger" id="message"></div></div>
    
    <div class="col-md-3"><?php echo $S['LABEL_NAME']; ?>:</div><div class="col-md-9"><input id="name" value="<?php echo $name ?>" name="name" type="text"/></div>
    <div class="col-md-3">Token: </div><div class="col-md-9"><input type="text" value="<?php echo $token; ?>" name="token" readonly="readonly"/></div>
    
    
    <div class="col-md-4 col-xs-12">
        <div class="box" id="target">
        </div>
    </div>
    <div class="col-md-8 col-xs-12">
        <div class="box" id="source">
        </div>
    </div>
    <div class="col-md-12">
        <button id="reset" type="button" class="btn">Reset</button>
        <input type="hidden" id="output_token" name="output_token">
        <button id="submit" type="submit" class="btn btn-primary" disabled="disabled">Finished</button>
    </div>
    </form>
</div>
    <?php 
} 
?>


<script>
    var items = <?php echo json_encode($data); ?>;
    var CSFS = {'items': items};
    console.log(CSFS);

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

    function create_item(item) {
        return $('<div role="button" class="col-md-3 col-xs-12 item"><span  class="glyphicon glyphicon-move handle" aria-hidden="true"></span> <span class="feature" id="' + item.No + '">' + item.Name + '</span> <span class="glyphicon glyphicon-question-sign" data-toggle="tooltip" title="' + item.Description + '"></span></div>');
    }

    function clear_all() {
        $('#output').val('');
        $('#source').empty();
        $('#target').empty();
        $('#submit').attr('disabled', 'disabled');
    }

    function reset() {
        clear_all();$
        $('#message').hide();

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
                    animation: 300,
                    handle: '.handle',
                    onAdd: onAdded,
                    onUpdate: onAdded,
                });
                
            $('[data-toggle="tooltip"]').tooltip();
                
    }


    (function () {
        <?php if($_SERVER['REQUEST_METHOD'] == 'GET'){ ?>
            reset();

            $('#reset').click(function () {
                reset();
            }); 
            $('#form').submit(onSubmit);
        <?php } ?>
    })();
    
    function is_form_valid(){
        var is_valid = false;
        is_valid = $('#name').val() != "" && is_count_correct(CSFS.items);
        return is_valid;
    }
    
    function onSubmit(e){
        if (is_form_valid()) { // is valid

            var list_ordered = get_ordered();
            var output = get_output_string(list_ordered);
            $('#output_token').val(output);
            return true;
        } 

        e.preventDefault();
        $('#message').text('Please fill all fields.').show();

        return false;  
    }

    function get_output_string(list_ordered) {
        var output = "";
        for (var i = 0; i < list_ordered.length; i++) {
            output += i;
            output += ":";
            output += list_ordered[i].id;
            if (i<list_ordered.length-1) {
                output += ",";
            }
        }
        output += "|";
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