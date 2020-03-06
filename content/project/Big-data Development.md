+++
data = 2018-01-01T00:00:00

title = "Analysis of operator traffic data based on Spark"

summary = "Operator traffic data based on big-data Development."

image_preview = ""

tags = ["Big-data Development"]

exterma_link = ""

math = false

# Optional external URL for project (replaces project detail page).

#url_code: ""
#url_pdf: ""
#url_slides: ""
#url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.

+++
• This project is mainly divided into three parts: building a cluster, processing log data, and storing data. First, set up a Spark distributed cluster on three Linux servers, one as the master and the other two as slaves. Start the hdfs and spark #programs respectively and upload the log data to the hdfs. The data synchronization is achieved through zookeeper. Use the java programming language to write spark programs to process log data, extract frequently accessed websites in the log, use a period of daily traffic and user mobile phone numbers, etc., filter the data according to conditions and store them in the distributed database Hbase.