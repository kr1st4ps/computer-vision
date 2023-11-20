#   Basket Counter
In this section I aim to train a detection model to detect basketballs and hoop rims, so that using the location of these objects I could count how many times balls go through the rim.

So far I have succesfully fine tuned YOLOv8 model that can detect basketballs and rims with pretty good accuracy, however the tracking fails:
-   when balls move to fast
-   background is too similar to balls
-   net of the rims moves
-   balls get behind the net

I need to fix these issues before moving to counting made shots.