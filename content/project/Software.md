+++
data = 2018-08-01T00:00:00

title = "Data scraping of the sunshine hotline administration platform"

summary = "Completing the Data scraping by using scrapy framework."

image_preview = "./ML2.jpeg"

tags = ["Software"]

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
• Use scrapy framework to crawl the number of the complaint post, the url of the post, the title of the post, and the content in the post. The process of scrapy crawling is divided into sending requests, parsing data, passing data to spiders, and storing data. First, send a request, transfer the requested data to the crawler, parse the returned data through xpath, submit the parsed URL to schedule, and transfer the #extracted content to the database. Send the request again through the schedule, access the content in the article, and perform page turning operations, and save the processed #data in the mongodb database.
