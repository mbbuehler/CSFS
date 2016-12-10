<!DOCTYPE html>
<html lang="en">
    <head>
        <?php
        require 'csfs_strings.php';
        ?>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0, user-scalable=no" />
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
        <!--<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css"/>-->
        <link rel="stylesheet" type="text/css" href="style.css"/>


    </head>
    <body>
        

        <?php
        
        require 'database.php';
        $dataset_name = $_GET['dataset_name'] ? filter_input(INPUT_GET, 'dataset_name', FILTER_SANITIZE_STRING) : 'student';
        $condition = $_GET['condition'] ? filter_input(INPUT_GET, 'condition', FILTER_SANITIZE_STRING) : 1;
        
        
        
        $S = get_S($condition);
//        var_dump($S);

        $data = get_data($dataset_name, $condition);
        $task = get_task('student', $condition);
        if (empty($data) || empty($task)) {
            die('No data found. Please use correct parameters for dataset_name and condition.');
        }

        $name = filter_input(INPUT_GET, 'name', FILTER_SANITIZE_STRING);
//    $token = filter_input(INPUT_GET, 'token', FILTER_SANITIZE_STRING);
        ?>

        <?php
        if ($_SERVER['REQUEST_METHOD'] === 'POST') {
//    var_dump($_POST);
            $name = filter_input(INPUT_POST, 'name', FILTER_SANITIZE_STRING);
            $result_token = handle_post($dataset_name, $condition);
//     $token = filter_input(INPUT_POST, 'token', FILTER_SANITIZE_STRING);
            ?>
            <h1><?php echo $S['THANKS']; ?>, <?php echo $name; ?>!</h1>
            <p><?php echo $S['TASK_AFTER_SUBMIT']; ?>
                <img width="50px" src="images/thumb-up.jpeg" title="saved"/>
            </p>

            <?php if($condition==1){ ?>
            <div class="col-md-3 col-xs-12">
                Token: 
            </div>
            <div class="col-md-9 col-xs-12">
                <input class="form-control" type="text" name="output_token" value="<?php echo $result_token; ?>" readonly="readonly">
            </div>
                    <?php 
            } ?>
            <?php
        } else {
            // Is not post
            ?>    
            <div class="container-fluid">
                <h1><?php echo $S['HEADER']; ?></h1>

                <div class="col-md-12">
                    <h2><?php echo $S['TASK_DESCRIPTION'] ?></h2>
                    <p> <?php echo str_replace('\\\\', '<br>', $task['description']); ?> </p>
                </div>
                <form method="post" id="form" action="">

                    <div class="col-md-12"><div class="alert alert-danger" id="message"></div></div>




                    <div class="col-md-12"><div class="form-group">
                            <label for="name"><?php echo $S['LABEL_NAME']; ?>:</label>
                            <input class="form-control" id="name" value="<?php echo $name ?>" name="name" type="text"/>
                        </div></div>
                    <div class="col-md-12"><div class="form-group">
                            <label for=""><?php echo $S['RANKHERE']; ?>:</label>

                        </div></div>

                    <div class="col-md-4 col-xs-6">
                        <div class="box" id="target">
                        </div>
                    </div>
                    <div class="col-md-8 col-xs-6">
                        <div class="box" id="source">
                        </div>
                    </div>
                    <div class="col-md-12 col-xs-12">
                        <input type="hidden" name="comment" id="comment"/>
                        <div class="form-group">
                            <label for="comment"><?php echo $S['LABEL_COMMENT']; ?>:</label>
                            <textarea class="form-control" rows="5" id="commentarea"></textarea>
                        </div>

                    </div>
                    <div class="col-md-12 col-xs-12">
                        <button id="reset" type="button" class="btn"><?php echo $S['RESET']; ?></button>
                        <input type="hidden" id="output_token" name="output_token">
                        <button id="submit" type="submit" class="btn btn-primary" disabled="disabled"><?php echo $S['FINISHED']; ?></button>
                    </div>
                </form>
            </div>
            <?php
        }
        ?>


        <script>
            var items = <?php echo json_encode($data); ?>;
            var CSFS = {'items': items};

            function prepend_numbering() {
                var $rows = $('#target').find('.row');
                for (var i = 0; i < $rows.length; i++) {
                    if ($rows.eq(i).find('.number').length == 0) {
                        $rows.eq(i).find('.feature').before($('<span class="number"></span>'));
                    }
                    console.log($rows.eq(i).find('span').eq(2).text());
                    $rows.eq(i).find('.number').text((i + 1) + '. ');
                }
                console.log('---');
            }

            function onAdded() {
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
                clear_all();
                $('#message').hide();

                for (var i = 0; i < CSFS.items.length; i++) {
                    var $item = create_item(CSFS.items[i]);
                    $('#source').append($item);
                }

                $('#source .item').dblclick(function () {
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
<?php if ($_SERVER['REQUEST_METHOD'] == 'GET') { 
    
    ?>
//            var isMobileOrTablet = mobileAndTabletcheck();
//            if(isMobileOrTablet){
//                window.alert('<?php echo $S['ONMOBILE']; ?>');
//            }
                    reset();

                    $('#reset').click(function () {
                        reset();
                    });
                    $('#form').submit(onSubmit);
<?php } ?>
            })();

            function is_form_valid() {
                var is_valid = false;
                is_valid = $('#name').val() != "" && is_count_correct(CSFS.items);
                return is_valid;
            }

            function onSubmit(e) {
                if (is_form_valid()) { // is valid

                    var list_ordered = get_ordered();
                    var output = get_output_string(list_ordered);
                    $('#output_token').val(output);
                    $('#comment').val($('#commentarea').val());
                    return true;
                }

                e.preventDefault();
                $('#message').text('<?php echo $S['FORMINVALID']; ?>').show();

                return false;
            }

            function get_output_string(list_ordered) {
                var output = "";
                for (var i = 0; i < list_ordered.length; i++) {
                    output += i;
                    output += ":";
                    output += list_ordered[i].id;
                    if (i < list_ordered.length - 1) {
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
            function mobileAndTabletcheck() {
  var check = false;
  (function(a){if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0,4))) check = true;})(navigator.userAgent||navigator.vendor||window.opera);
  return check;
            }
        </script>
    </body>
</html>