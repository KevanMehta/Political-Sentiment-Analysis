# Big data Project

## Installation Guide Ubuntu

### Java installation
`sudo apt-get install openjdk-8-jdk`


### Spark Installation
Download spark from the link https://www.apache.org/dyn/closer.lua/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz. 
And you can add it PATH if needed.

### Kafka Installation
Download the kafka from https://www.apache.org/dyn/closer.cgi?path=/kafka/3.4.0/kafka_2.13-3.4.0.tgz. And you can add it PATH if needed.

### Elastic Search and Kibana Installation
Add the repo first. Add PGP key
`wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -`

Install apt-transport-https

`sudo apt-get install apt-transport-https`

Finally add the repo

`echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee –a /etc/apt/sources.list.d/elastic-7.x.list`

Install elasticsearch

`sudo apt-get update`

`sudo apt-get install elasticsearch`

configure elastic search

`sudo nano /etc/elasticsearch/elasticsearch.yml`
uncomment network.host and network.port. Since I run this on AWS, I have set the `network.host:0.0.0.0` and `port:9200`, `discovery.type: single-node`.

Additionally, the jvm heap size can be increased. I have configured a 8GB instance and have set the heap size to 4G.

```
-Xms4g
-Xmx4g
```
Enable and start the elasticsearch
```
sudo systemctl start elasticsearch.service
sudo systemctl enable elasticsearch.service
```

Install Kibana
`sudo apt-get install kibana`
Configure kibana
`sudo nano /etc/kibana/kibana.yml`

uncomment network.host and network.port. Since I run this on AWS, I have set the `server.host:0.0.0.0` and `port:560`, `elasticsearch.hosts: ["http://0.0.0.0:9200"]`

enable and start kibana
```
sudo systemctl start kibana
sudo systemctl start kibana
```

### Logstash Installation
```
sudo systemctl start logstash
sudo systemctl enable logstash
```

### Commands to run


1. Run zookeeper first with below commands
    
    ```
    zookeeper-server-start.sh /opt/kafka/kafka_2.13-3.4.0/config/zookeeper.properties
    ```
2. Once zookeeper is up, start the kafka server 
    ```
    kafka-server-start.sh /opt/kafka/kafka_2.13-3.4.0/config/server.properties
    ```

6. Run logstash that would publish it to elastic search
```
sudo /usr/share/logstash/bin/logstash -f logstash_ne.conf
```

The contents of logstash_ne.conf looks like below. The ek stack is installed on AWS by me and spark, kafka and logstash runs locally due to memory and space constraints.

```
input {
  http {
    host => "localhost"
    port => 8888
  }
}

output {
  elasticsearch {
    hosts => ["http://<AWS_IP>:9200"] #Points to my AWS server as the memory is limited
    index => "named_entities_index"
  }
}
```

_Note: All the paths are added to path in bashrc_

## How to run the code?

### APIS
The code connects to different API

* NewsAPI
* NewsAPI.ai
* Newsapi.org


The secrets are shared too.


### LSTM
The LSTM code runs on Google Colab and needs GPU enabled for better processing

### KNN and Naive Bayes
The KNN codes for training and inference are published in the below links

  KnnInference
  https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8602585654301424/335775362613554/8141684773018690/latest.html

  KnnTraining
  https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8602585654301424/3170926744466459/8141684773018690/latest.html

  NaviesBayes Training 
  https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8602585654301424/1275282579552992/8141684773018690/latest.html

  NavieBayes Inference
  https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8602585654301424/4195266324204897/8141684773018690/latest.html

The codes can be run on DataBricks Cluster
