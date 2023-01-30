CREATE TABLE logging (
    projid VARCHAR(250),
    vid CHAR(50),
    tstamp TIMESTAMP,
    epoch int,
    step int,
    name VARCHAR(150),
    value VARCHAR(250)
);
COPY logging(projid, vid, tstamp, epoch, step, name, value)
FROM '/tmp/dump/logging_12.csv' DELIMiTER ',' CSV HEADER;