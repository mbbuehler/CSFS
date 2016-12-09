<?php

/* 
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

require 'local.php';
   
function get_connection(){
    $conn = mysqli_connect(DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE);

    // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }
    return $conn;
}

/***
 * Fetches all data from a table given name and condition.
 */
function get_data($dataset_name, $condition){
    
    $table_name = $dataset_name."_".$condition; // e.g. student_1
    
    $conn = get_connection();
    $sql = "SELECT * FROM ".$table_name;
    $result = $conn->query($sql);
    $data = result_to_array($result, 'shuffle');
    return $data;
}

/***
 * Converts a $conn->query($sql) into an array.
 */
function result_to_array($result, $do_shuffle=False){
    $data = array();

    if ($result->num_rows > 0) {
        // output data of each row
        while($row = $result->fetch_assoc()) {
            $data[] = $row;
        }
    } else {
//        echo "0 results";
    }
    if($do_shuffle=='shuffle'){
        shuffle($data); // randomise order so the crowd has no bias
    }
    return $data;
}

/***
 * Gets the task description
 */
function get_task($dataset_name, $condition){
    $conn = get_connection();
    $sql = "SELECT * FROM `tasks` WHERE `dataset_name` LIKE '".$dataset_name."' AND `condition` LIKE '".$condition."'";
    $result = $conn->query($sql);
    $conn->close();
    $data = result_to_array($result);
    $data = $data[0]; // this should only return one row
    return $data;
}

function handle_post($dataset_name, $condition){
    $clean = [];
    foreach ($_POST as $key=>$val){
        $clean[$key] = filter_input(INPUT_POST, $key, FILTER_SANITIZE_STRING);
    }
    
    $output_token = save_ranking($dataset_name, $condition, $clean);
    
    if($output_token == false){
        return "Something went wrong. Please try again.";
    }
    return $output_token;
    
}

function save_ranking($dataset_name, $condition, $data){
    $name = $data['name'];
    $output_token = $data['output_token'];
    
    $hash = md5($name);
    $output_token .= $hash;
    
    $comment = $data['comment'];
    
    $conn = get_connection();
    $sql = "INSERT INTO result (dataset_name, cond, name, output_token, comment)
VALUES ('".$dataset_name."', '".$condition."', '".$name."','".$output_token."', '".$comment."')";
    
    if ($conn->query($sql) === TRUE) {
        
    return $output_token;
} else {
    echo "Error: " . $sql . "<br>" . $conn->error;
}

$conn->close();
return false;
    
}
