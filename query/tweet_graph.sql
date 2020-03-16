SELECT
	message_id_x AS source,
	message_id_y AS target,
	COUNT ( word ) AS weight
FROM
	[dbo].[allkeywords]
WHERE
	message_id_x <> message_id_y
GROUP BY
	message_id_x,
	message_id_y