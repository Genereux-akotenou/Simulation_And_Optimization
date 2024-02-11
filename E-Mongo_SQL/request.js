db.users.find({"movies.movieid": 1196}).count()
// 2990

db.users.find({"movies.movieid": { $all: [260, 1196, 1210]} }).count()
// 1926

db.users.find({"movies": { $size: 48 }}).count()
// 51

db.users.updateMany({}, [{$set: {"num_ratings": {$size: "$movies"}}}])
//db.users.aggregate([{ $set: {"num_ratings": {$size: "$movies"}}}])

db.users.find({"num_ratings": {$gt: 90} }).count()
// 3114

db.users.find({ 
    "movies.timestamp": {
        $gt: Math.round(new Date("2001-01-01").getTime() / 1000)
    }
}).count()
// 1177
db.users.aggregate([
    { $unwind: "$movies" }, 
    { 
        $match: {
            "movies.timestamp": {
                $gt: Math.round(new Date("2001-01-01").getTime() / 1000)
            } 
        } 
    }, 
    { $count: "count" }
])
// {
//   count: 95453
// }

// db.users.findOne({name: "Jayson Brad"}).movies.slice(-3)
// [
//     { movieid: 1097, rating: 3, timestamp: 956776340 },
//     { movieid: 2043, rating: 2, timestamp: 956778658 },
//     { movieid: 3783, rating: 4, timestamp: 963610480 }
// ]

db.users.aggregate([
    { $match: { "name": "Jayson Brad" } },
    { $unwind: "$movies" },
    { $sort: { "movies.timestamp": -1 } },
    { $limit: 3 },
    { $project: { _id: 0, "movieid": "$movies.movieid", "rating": "$movies.rating", "timestamp": "$movies.timestamp" } }
])
// {
//     movieid: 3639,
//     rating: 2,
//     timestamp: 995664224
// }
// {
//     movieid: 2347,
//     rating: 3,
//     timestamp: 995664198
// }
// {
//     movieid: 3635,
//     rating: 3,
//     timestamp: 995664198
// }

db.users.aggregate([
    {
        $unwind: "$movies"
    },
    {
        $match: {
            "name": "Tracy Edward", 
            "movies.movieid": 1210
        }
    },
    {
        $project: {
            "movies.movieid": 0,
            "movies.timestamp": 0,
        }
    }
])
// {
//     _id: 5951,
//     name: 'Tracy Edward',
//     gender: 'M',
//     age: 24,
//     occupation: 'college/grad student',
//     movies: {
//       rating: 5
//     },
//     num_ratings: 53
// }

db.users.aggregate([
    {
        $unwind: "$movies"
    },
    {
        $match: {
            "movies.rating": 5
        }
    },
    {
        $lookup: {
            from: "movies",
            localField: "movies.movieid",
            foreignField: "_id",
            as: "movieDetails"
        }
    },
    {
        $unwind: "$movieDetails"
    },
    {
        $match: {
            "movieDetails.title": "Untouchables, The (1987)"
        }
    },
    {
        $count: "number_count"
    }
])
// {
//     number_count: 317
// }


// Ecriture 
// -------
//  1. L’utilisateur Barry Erin vient juste de voir le film Nixon, qui a pour id 14 ; il lui attribue la
// note de 4. Mettre à jour la base de données pour prendre en compte cette note. N’oubliez
// pas que le champ num_rattings doit représenter le nombre de films notés par un utilisateur.
// (update)
// -------

db.users.updateOne(
    {
          name: "Barry Erin"
    }, 
    {
        $inc: {
            "num_ratings": 1
        },
        $push: {
            "movies": {
                "movieid": 14,
                "rating": 4,
                "timestamp": Math.round(new Date().getTime() / 1000)
            }
        }
    }
)
// {
//     acknowledged: true,
//     insertedId: null,
//     matchedCount: 1,
//     modifiedCount: 1,
//     upsertedCount: 0
// }


db.users.updateOne(
    {
        "name": "Marquis Billie"
    },
    {
        $inc: {
            "num_ratings": -1
        },
        $pull: {
            "movies": {
                "movieid": 1311
            }
        }
    }
)
// {
//     acknowledged: true,
//     insertedId: null,
//     matchedCount: 1,
//     modifiedCount: 1,
//     upsertedCount: 0
// }


db.movies.updateOne(
    {
        title: "Cinderella (1950)"
    },
    {
        $set: {
            "genres": "Animation|Children's|Musical"
        }
    }
)
// {
//     acknowledged: true,
//     insertedId: null,
//     matchedCount: 1,
//     modifiedCount: 1,
//     upsertedCount: 0
// }