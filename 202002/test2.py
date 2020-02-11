html = """
<html>
   <head>
      <meta charset="utf-8">
   </head>

   <body>
       <font color=red> @out</font>
   </body>
</html>
"""

html = html.replace("@out",  "제목출력")
#template engine에서 실시간으로 바꿔준다.

print(html)