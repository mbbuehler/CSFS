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
        <script src="ranking.js"></script>
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
        $task = get_task($dataset_name, $condition);
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
            <h1><?php echo $S['THANKS']; ?>!</h1>
            <p><?php echo $S['TASK_AFTER_SUBMIT']; ?>
                <img width="50px" src="images/thumb-up.jpeg" title="saved"/>
            </p>

            <?php if ($condition == 1 || $condition == 3) { ?>
                <div class="col-md-3 col-xs-12">
                    Token: 
                </div>
                <div class="col-md-9 col-xs-12">
                    <input class="form-control" type="text" name="output_token" value="<?php echo $result_token; ?>" readonly="readonly">
                </div>
                <?php }
            ?>
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
                            <textarea class="form-control" rows="5" id="commentarea" <?php if($condition==2) { echo 'placeholder="Hast du beruflich / in deiner Freizeit mit dem Thema zu tun?"';}?>></textarea>
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
        </script>
        <footer id="footer">
            <hr>
            <p><a href="http://mbuehler.ch">Marcel B&uuml;hler</a>, <?php echo date("Y"); ?></p>
        </footer>

    </body>
</html>