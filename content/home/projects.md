+++
# Accomplishments widget.
widget = "accomplishments"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true  # Activate this widget? true/false
weight = 65  # Order that this section will appear.

title = "Projects"
subtitle = ""

# Date format
#   Refer to https://sourcethemes.com/academic/docs/customization/#date-format
date_format = "Jan 2006"

#[[item]]
#  title = "Analysis of operator traffic data based on Spark"
#  date_start = "2016-11-01"
#  date_end = "2017-05-01"
#  description = "This project is mainly divided into three parts: building a cluster, #processing log data, and storing data. First, set up a Spark distributed cluster on three #Linux servers, one as the master and the other two as slaves. Start the hdfs and spark #programs respectively and upload the log data to the hdfs. The data synchronization is #achieved through zookeeper. Use the java programming language to write spark programs to #process log data, extract frequently accessed websites in the log, use a period of daily #traffic and user mobile phone numbers, etc., filter the data according to conditions and #store them in the distributed database Hbase."

#[[item]]
#  title = "Data scraping of the sunshine hotline administration platform based on scrapy #framework"
#  date_start = "2019-03-01"
#  date_end = "2019-07-01"
#  description = "Use scrapy framework to crawl the number of the complaint post, the url #of the post, the title of the post, and the content in the post. The process of scrapy #crawling is divided into sending requests, parsing data, passing data to spiders, and #storing data. First, send a request, transfer the requested data to the crawler, parse #the returned data through xpath, submit the parsed URL to schedule, and transfer the #extracted content to the database. Send the request again through the schedule, access #the content in the article, and perform page turning operations, and save the processed #data in the mongodb database."
+++



[content]
  # Page type to display. E.g. project.
  page_type = "project"
  
  # Filter toolbar (optional).
  # Add or remove as many filters (`[[content.filter_button]]` instances) as you like.
  # To show all items, set `tag` to "*".
  # To filter by a specific tag, set `tag` to an existing tag name.
  # To remove toolbar, delete/comment all instances of `[[content.filter_button]]` below.
  
  # Default filter index (e.g. 0 corresponds to the first `[[filter_button]]` instance below).
  filter_default = 0
  
  [[content.filter_button]]
   name = "All"
   tag = "*"
  
  [[content.filter_button]]
   name = "Deep Learning"
   tag = "Deep Learning"
  
  [[content.filter_button]]
   name = "Software Development"
   tag = "Software"

[design]
  # Choose how many columns the section has. Valid values: 1 or 2.
  columns = "2"

  # Toggle between the various page layout types.
  #   1 = List
  #   2 = Compact
  #   3 = Card
  #   5 = Showcase
  view = 3

  # For Showcase view, flip alternate rows?
  flip_alt_rows = false

[design.background]
  
  
[advanced]
 # Custom CSS. 
 css_style = ""
 
 # CSS class.
 css_class = ""
+++






