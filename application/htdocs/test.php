<html lang="en">
    <head>
        <?php
        require 'csfs_strings.php';
        ?>
        <meta charset="UTF-8">
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

<ul id="source">
    <li class="item">test1</li>
    <li class="item">test2</li>
    <li class="item">test3</li>
    <li class="item">test4</li>
</ul>

<script>
    $(document).ready(function(){
        var source = document.getElementById('source');
    Sortable.create(
                        source,
                        {
                            group: "ranking-source",
                            draggable: '.item',
                            animation: 200,
                        }
                );
        });
    
    </script>
    </body>