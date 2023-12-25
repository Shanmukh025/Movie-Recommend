# Movie Recommendation Algorithm using Machine Learning.

This is a movie recommendation system using Non-Negative Regularized Matrix Factorization.

1.  In a recommendation system such as Netflix and Prime Video, there is a group of users and a set of items (movies for the above two systems). Given that each user has rated some items in the system, we would like to predict how the users would rate the items they have not yet rated so that we can make recommendations to the users.

2.  I implemented Regularized Matrix factorization based on the following :

    1. Movie Lens dataset, describes a 5-star rating. It contains 21,000,000 ratings and 510,000 tag applications applied to 30,000 movies by 230,000 users.
    2. Ratings.csv file is the rating file. Each line of this file after the header row represents one rating of one movie by one user and has the following format: userId, movieId, rating, and timestamps.
    3. Movie information is contained in the file movies.csv. Each line of this file after the header row represents one movie and has the following format: movieId, title, genres.

3.  MatrixFactorization.py codes implement the regularized matrix factorization model(Gradient Descent) without using any Machine Learning libraries and save it as a file, which can then be used to perform recommendations. I used the below parameter values:

    1. steps: the maximum number of steps to perform the optimization was set to 5000
    2. alpha: the learning rate was set to 0.0002
    3. beta: the regularization parameter was set to 0.02
    4. k: hidden latent features were set to 8

4.  Recommend.py takes in users as input and recommends the top 25 movies out of his/her unrated movies if the predicted rating for these movies is greater than 3.5 as ideally we want to recommend only good movies

5.  Results for users 1 to 5 as follows:

            Top 25 movie recommendations for the "User 1"

                Killer's Kiss (1955)
                Liberty Heights (1999)
                Angel at My Table  An (1990)
                Scout  The (1994)
                Great Santini  The (1979)
                Pet Sematary (1989)
                Goal! The Dream Begins (Goal!) (2005)
                Moscow on the Hudson (1984)
                Innerspace (1987)
                Ballistic: Ecks vs. Sever (2002)
                Doc Hollywood (1991)
                Heathers (1989)
                One Flew Over the Cuckoo's Nest (1975)
                Kingdom  The (Riget) (1994)
                Game Over: Kasparov and the Machine (2003)
                I Know What You Did Last Summer (1997)
                Supercop (Police Story 3: Supercop) (Jing cha gu shi III: Chao ji jing cha) (1992)
                Amadeus (1984)
                Vibes (1988)
                Ghost and Mrs. Muir  The (1947)
                Amazing Grace (2006)
                Downloaded (2013)
                Waking Ned Devine (a.k.a. Waking Ned) (1998)
                It Happened One Night (1934)
                City of Angels (1998)


            Top 25 movie recommendations for the "User 2"

                9 (2009)
                Pet Sematary (1989)
                Innerspace (1987)
                Before the Rain (Pred dozhdot) (1994)
                Ballistic: Ecks vs. Sever (2002)
                Angel at My Table  An (1990)
                Nanook of the North (1922)
                Mighty Aphrodite (1995)
                Supercop (Police Story 3: Supercop) (Jing cha gu shi III: Chao ji jing cha) (1992)
                Killer's Kiss (1955)
                Selma (2014)
                Gone Girl (2014)
                Charlie Wilson's War (2007)
                Downloaded (2013)
                Divine Intervention (Yadon ilaheyya) (2002)
                Thief of Bagdad  The (1940)
                Doc Hollywood (1991)
                King of Kings (1961)
                Heartbreakers (2001)
                Made in U.S.A. (1966)
                Great Santini  The (1979)
                Together (Han ni Zai Yiki) (2002)
                Dangerous Liaisons (1988)
                Thinner (1996)
                Not Another Teen Movie (2001)


            Top 25 movie recommendations for the "User 3"

                Killer's Kiss (1955)
                My Life as a Dog (Mitt liv som hund) (1985)
                9 (2009)
                White Sound  The (Das weiÌÙe Rauschen) (2001)
                Scout  The (1994)
                Spellbound (1945)
                Bride of the Monster (1955)
                West Side Story (1961)
                Ballistic: Ecks vs. Sever (2002)
                One Flew Over the Cuckoo's Nest (1975)
                Game Over: Kasparov and the Machine (2003)
                Supercop (Police Story 3: Supercop) (Jing cha gu shi III: Chao ji jing cha) (1992)
                Prime Suspect (1991)
                Dangerous Liaisons (1988)
                Hunt  The (Jagten) (2012)
                Great Expectations (1998)
                Freejack (1992)
                Full Metal Jacket (1987)
                Gentleman's Agreement (1947)
                Innerspace (1987)
                Heartbreakers (2001)
                Moscow on the Hudson (1984)
                Just Go with It (2011)
                Thomas and the Magic Railroad (2000)
                In the Heat of the Night (1967)


            Top 25 movie recommendations for the "User 4"

                Pet Sematary (1989)
                9 (2009)
                Angel at My Table  An (1990)
                Ballistic: Ecks vs. Sever (2002)
                Kingdom  The (Riget) (1994)
                Mighty Aphrodite (1995)
                Together (Han ni Zai Yiki) (2002)
                Thief of Bagdad  The (1940)
                Divine Intervention (Yadon ilaheyya) (2002)
                Star Wars: Episode VI - Return of the Jedi (1983)
                Downloaded (2013)
                Red Lights (Feux rouges) (2004)
                Vibes (1988)
                Moscow on the Hudson (1984)
                Killer's Kiss (1955)
                Innerspace (1987)
                Full Metal Jacket (1987)
                Trials of Henry Kissinger  The (2002)
                Doc Hollywood (1991)
                Supercop (Police Story 3: Supercop) (Jing cha gu shi III: Chao ji jing cha) (1992)
                Desperado (1995)
                Nanook of the North (1922)
                Not Another Teen Movie (2001)
                Waking Ned Devine (a.k.a. Waking Ned) (1998)
                Ichi the Killer (Koroshiya 1) (2001)


            Top 25 movie recommendations for the "User 5"

                Pet Sematary (1989)
                Angel at My Table  An (1990)
                9 (2009)
                Gone Girl (2014)
                Nanook of the North (1922)
                Ballistic: Ecks vs. Sever (2002)
                Downloaded (2013)
                Before the Rain (Pred dozhdot) (1994)
                Divine Intervention (Yadon ilaheyya) (2002)
                Kingdom  The (Riget) (1994)
                Amazing Grace (2006)
                Great Expectations (1998)
                Martian Child (2007)
                Waking Ned Devine (a.k.a. Waking Ned) (1998)
                Vibes (1988)
                Au Hasard Balthazar (1966)
                King of Kings (1961)
                Trials of Henry Kissinger  The (2002)
                Dead Poets Society (1989)
                Charlie Wilson's War (2007)
                Omega Man  The (1971)
                Full Metal Jacket (1987)
                Thinner (1996)
                Doc Hollywood (1991)
                Killer's Kiss (1955)
