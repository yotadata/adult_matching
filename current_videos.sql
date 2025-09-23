SET session_replication_role = replica;

--
-- PostgreSQL database dump
--

-- Dumped from database version 17.4
-- Dumped by pg_dump version 17.4

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Data for Name: audit_log_entries; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--

INSERT INTO "auth"."audit_log_entries" ("instance_id", "id", "payload", "created_at", "ip_address") VALUES
	('00000000-0000-0000-0000-000000000000', '1727c8e8-f9b2-4cd3-ace1-7c0fe0f47cab', '{"action":"user_signedup","actor_id":"e010209e-1e56-450d-bf5f-76a478a9c6ca","actor_username":"test@example.com","actor_via_sso":false,"log_type":"team","traits":{"provider":"email"}}', '2025-08-23 17:32:36.013986+00', ''),
	('00000000-0000-0000-0000-000000000000', '8fae64c7-1e57-40f7-a679-629fc1f533be', '{"action":"login","actor_id":"e010209e-1e56-450d-bf5f-76a478a9c6ca","actor_username":"test@example.com","actor_via_sso":false,"log_type":"account","traits":{"provider":"email"}}', '2025-08-23 17:32:36.018831+00', ''),
	('00000000-0000-0000-0000-000000000000', 'a2cf7306-7ecc-4101-b7f5-10a715aa049f', '{"action":"login","actor_id":"e010209e-1e56-450d-bf5f-76a478a9c6ca","actor_username":"test@example.com","actor_via_sso":false,"log_type":"account","traits":{"provider":"email"}}', '2025-08-24 03:28:35.016927+00', ''),
	('00000000-0000-0000-0000-000000000000', 'ba46e57c-cd2b-4a70-95ba-ff3adde2ab97', '{"action":"user_repeated_signup","actor_id":"e010209e-1e56-450d-bf5f-76a478a9c6ca","actor_username":"test@example.com","actor_via_sso":false,"log_type":"user","traits":{"provider":"email"}}', '2025-09-02 13:13:13.684906+00', ''),
	('00000000-0000-0000-0000-000000000000', '753b8ca8-3f8e-41ec-b4c1-0e4817de3720', '{"action":"login","actor_id":"e010209e-1e56-450d-bf5f-76a478a9c6ca","actor_username":"test@example.com","actor_via_sso":false,"log_type":"account","traits":{"provider":"email"}}', '2025-09-02 13:13:23.437255+00', '');


--
-- Data for Name: flow_state; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--



--
-- Data for Name: users; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--

INSERT INTO "auth"."users" ("instance_id", "id", "aud", "role", "email", "encrypted_password", "email_confirmed_at", "invited_at", "confirmation_token", "confirmation_sent_at", "recovery_token", "recovery_sent_at", "email_change_token_new", "email_change", "email_change_sent_at", "last_sign_in_at", "raw_app_meta_data", "raw_user_meta_data", "is_super_admin", "created_at", "updated_at", "phone", "phone_confirmed_at", "phone_change", "phone_change_token", "phone_change_sent_at", "email_change_token_current", "email_change_confirm_status", "banned_until", "reauthentication_token", "reauthentication_sent_at", "is_sso_user", "deleted_at", "is_anonymous") VALUES
	('00000000-0000-0000-0000-000000000000', 'e010209e-1e56-450d-bf5f-76a478a9c6ca', 'authenticated', 'authenticated', 'test@example.com', '$2a$10$v1Aflbkrkhwe9x4e19m2oewWMoTSaI3cW1rDLDzYurQpyckQayYlm', '2025-08-23 17:32:36.015381+00', NULL, '', NULL, '', NULL, '', '', NULL, '2025-09-02 13:13:23.463647+00', '{"provider": "email", "providers": ["email"]}', '{"sub": "e010209e-1e56-450d-bf5f-76a478a9c6ca", "email": "test@example.com", "email_verified": true, "phone_verified": false}', NULL, '2025-08-23 17:32:36.001891+00', '2025-09-02 13:13:23.478691+00', NULL, NULL, '', '', NULL, '', 0, NULL, '', NULL, false, NULL, false);


--
-- Data for Name: identities; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--

INSERT INTO "auth"."identities" ("provider_id", "user_id", "identity_data", "provider", "last_sign_in_at", "created_at", "updated_at", "id") VALUES
	('e010209e-1e56-450d-bf5f-76a478a9c6ca', 'e010209e-1e56-450d-bf5f-76a478a9c6ca', '{"sub": "e010209e-1e56-450d-bf5f-76a478a9c6ca", "email": "test@example.com", "email_verified": false, "phone_verified": false}', 'email', '2025-08-23 17:32:36.010547+00', '2025-08-23 17:32:36.010595+00', '2025-08-23 17:32:36.010595+00', 'da4b2f04-2e8b-4804-a32d-b8fef2cee0b6');


--
-- Data for Name: instances; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--



--
-- Data for Name: sessions; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--

INSERT INTO "auth"."sessions" ("id", "user_id", "created_at", "updated_at", "factor_id", "aal", "not_after", "refreshed_at", "user_agent", "ip", "tag") VALUES
	('f64d38cf-cfb6-48a4-be32-17bdc8a4d296', 'e010209e-1e56-450d-bf5f-76a478a9c6ca', '2025-08-23 17:32:36.019752+00', '2025-08-23 17:32:36.019752+00', NULL, 'aal1', NULL, NULL, 'curl/8.5.0', '172.19.0.1', NULL),
	('4f00c424-4694-42fa-a7b6-67d89f24a456', 'e010209e-1e56-450d-bf5f-76a478a9c6ca', '2025-08-24 03:28:35.019424+00', '2025-08-24 03:28:35.019424+00', NULL, 'aal1', NULL, NULL, 'curl/8.5.0', '172.19.0.1', NULL),
	('a1c107f6-9d85-43ba-80fc-ceb8b68d21f9', 'e010209e-1e56-450d-bf5f-76a478a9c6ca', '2025-09-02 13:13:23.464339+00', '2025-09-02 13:13:23.464339+00', NULL, 'aal1', NULL, NULL, 'curl/8.5.0', '172.19.0.1', NULL);


--
-- Data for Name: mfa_amr_claims; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--

INSERT INTO "auth"."mfa_amr_claims" ("session_id", "created_at", "updated_at", "authentication_method", "id") VALUES
	('f64d38cf-cfb6-48a4-be32-17bdc8a4d296', '2025-08-23 17:32:36.02437+00', '2025-08-23 17:32:36.02437+00', 'password', 'b43e3d50-5797-49ce-88c4-ebd2c6b5c855'),
	('4f00c424-4694-42fa-a7b6-67d89f24a456', '2025-08-24 03:28:35.024023+00', '2025-08-24 03:28:35.024023+00', 'password', 'a82643f8-895d-44c4-820e-4a71c53d77bd'),
	('a1c107f6-9d85-43ba-80fc-ceb8b68d21f9', '2025-09-02 13:13:23.480293+00', '2025-09-02 13:13:23.480293+00', 'password', '575a0670-c8c3-4955-8415-3a7261f4e2f0');


--
-- Data for Name: mfa_factors; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--



--
-- Data for Name: mfa_challenges; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--



--
-- Data for Name: one_time_tokens; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--



--
-- Data for Name: refresh_tokens; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--

INSERT INTO "auth"."refresh_tokens" ("instance_id", "id", "token", "user_id", "revoked", "created_at", "updated_at", "parent", "session_id") VALUES
	('00000000-0000-0000-0000-000000000000', 1, 'hedkarbtn3dc', 'e010209e-1e56-450d-bf5f-76a478a9c6ca', false, '2025-08-23 17:32:36.021553+00', '2025-08-23 17:32:36.021553+00', NULL, 'f64d38cf-cfb6-48a4-be32-17bdc8a4d296'),
	('00000000-0000-0000-0000-000000000000', 2, 'fgnv4ac2xw54', 'e010209e-1e56-450d-bf5f-76a478a9c6ca', false, '2025-08-24 03:28:35.021122+00', '2025-08-24 03:28:35.021122+00', NULL, '4f00c424-4694-42fa-a7b6-67d89f24a456'),
	('00000000-0000-0000-0000-000000000000', 3, 'egoc7e3tv5pm', 'e010209e-1e56-450d-bf5f-76a478a9c6ca', false, '2025-09-02 13:13:23.4707+00', '2025-09-02 13:13:23.4707+00', NULL, 'a1c107f6-9d85-43ba-80fc-ceb8b68d21f9');


--
-- Data for Name: sso_providers; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--



--
-- Data for Name: saml_providers; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--



--
-- Data for Name: saml_relay_states; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--



--
-- Data for Name: sso_domains; Type: TABLE DATA; Schema: auth; Owner: supabase_auth_admin
--



--
-- Data for Name: videos; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO "public"."videos" ("id", "external_id", "title", "description", "duration_seconds", "thumbnail_url", "preview_video_url", "distribution_code", "maker_code", "director", "series", "maker", "label", "genre", "price", "distribution_started_at", "product_released_at", "sample_video_url", "image_urls", "source", "published_at", "created_at") VALUES
	('550e8400-e29b-41d4-a716-446655441000', 'TEST001', 'テスト動画1', 'これはテスト用の動画です', 3600, 'https://example.com/thumb1.jpg', 'https://example.com/preview1.mp4', NULL, NULL, NULL, NULL, 'テストメーカーA', NULL, 'ドラマ', 2980, NULL, NULL, 'https://example.com/sample1.mp4', '{https://example.com/img1.jpg}', 'test', '2025-08-23 17:31:13.176783+00', '2025-08-23 17:31:13.176783+00'),
	('550e8400-e29b-41d4-a716-446655441001', 'TEST002', 'テスト動画2', 'これもテスト用の動画です', 2700, 'https://example.com/thumb2.jpg', 'https://example.com/preview2.mp4', NULL, NULL, NULL, NULL, 'テストメーカーB', NULL, 'コメディ', 1980, NULL, NULL, 'https://example.com/sample2.mp4', '{https://example.com/img2.jpg}', 'test', '2025-08-23 17:31:13.176783+00', '2025-08-23 17:31:13.176783+00'),
	('550e8400-e29b-41d4-a716-446655441002', 'TEST003', 'テスト動画3', '3番目のテスト動画', 4200, 'https://example.com/thumb3.jpg', 'https://example.com/preview3.mp4', NULL, NULL, NULL, NULL, 'テストメーカーC', NULL, 'アクション', 3980, NULL, NULL, 'https://example.com/sample3.mp4', '{https://example.com/img3.jpg}', 'test', '2025-08-23 17:31:13.176783+00', '2025-08-23 17:31:13.176783+00'),
	('550e8400-e29b-41d4-a716-446655441003', 'TEST004', 'テスト動画4', '4番目のテスト動画', 3300, 'https://example.com/thumb4.jpg', 'https://example.com/preview4.mp4', NULL, NULL, NULL, NULL, 'テストメーカーA', NULL, 'ドラマ', 2480, NULL, NULL, 'https://example.com/sample4.mp4', '{https://example.com/img4.jpg}', 'test', '2025-08-23 17:31:13.176783+00', '2025-08-23 17:31:13.176783+00'),
	('550e8400-e29b-41d4-a716-446655441004', 'TEST005', 'テスト動画5', '5番目のテスト動画', 2900, 'https://example.com/thumb5.jpg', 'https://example.com/preview5.mp4', NULL, NULL, NULL, NULL, 'テストメーカーB', NULL, 'コメディ', 1780, NULL, NULL, 'https://example.com/sample5.mp4', '{https://example.com/img5.jpg}', 'test', '2025-08-23 17:31:13.176783+00', '2025-08-23 17:31:13.176783+00'),
	('87fd544b-024c-441a-828d-4cd920ecd138', 'test001', 'テストビデオ1', 'これはテスト用の動画です', NULL, 'https://example.com/images/test1_large.jpg', 'https://example.com/movies/test1_720.mp4', 'test001', NULL, 'テスト監督', 'テストシリーズ', 'テスト制作会社', '', 'テストジャンル', 1000, '2025-08-24 06:28:07.138+00', '2025-08-24 06:28:07.138+00', 'https://example.com/movies/test1_720.mp4', '{https://example.com/samples/test1_1.jpg}', 'dmm', '2025-08-24 06:28:07.138+00', '2025-08-24 06:28:07.278227+00'),
	('0aa57ba7-b627-45b4-b01d-85a4c4ce0e2b', 'api_video_001', 'API取得動画サンプル1', 'これはDMM API経由で取得された動画1です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_001/api_video_001pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_001/api_video_001_mhb_w.mp4', 'api_video_001', NULL, '監督1', 'シリーズ1', '制作会社A', '', 'ドラマ', 2000, '2025-09-06 05:16:48.411+00', '2025-09-06 05:16:48.411+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_001/api_video_001_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_001/api_video_001jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_001/api_video_001jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_001/api_video_001jp-3.jpg}', 'dmm', '2025-09-06 05:16:48.411+00', '2025-09-06 05:16:48.517992+00'),
	('21a62476-e985-4cbd-b976-8f50d78c0b50', 'api_video_002', 'API取得動画サンプル2', 'これはDMM API経由で取得された動画2です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_002/api_video_002pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_002/api_video_002_mhb_w.mp4', 'api_video_002', NULL, '監督2', 'シリーズ2', '制作会社B', '', 'コメディ', 2500, '2025-09-05 05:16:48.412+00', '2025-09-05 05:16:48.412+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_002/api_video_002_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_002/api_video_002jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_002/api_video_002jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_002/api_video_002jp-3.jpg}', 'dmm', '2025-09-05 05:16:48.412+00', '2025-09-06 05:16:48.640689+00'),
	('b4d6190b-4f08-410a-8759-15fd7d732b4b', 'api_video_003', 'API取得動画サンプル3', 'これはDMM API経由で取得された動画3です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_003/api_video_003pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_003/api_video_003_mhb_w.mp4', 'api_video_003', NULL, '監督3', 'シリーズ3', '制作会社C', '', 'アクション', 3000, '2025-09-04 05:16:48.412+00', '2025-09-04 05:16:48.412+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_003/api_video_003_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_003/api_video_003jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_003/api_video_003jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_003/api_video_003jp-3.jpg}', 'dmm', '2025-09-04 05:16:48.412+00', '2025-09-06 05:16:48.720249+00'),
	('740ca061-4a04-452c-9a8c-0037363125c8', 'api_video_004', 'API取得動画サンプル4', 'これはDMM API経由で取得された動画4です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_004/api_video_004pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_004/api_video_004_mhb_w.mp4', 'api_video_004', NULL, '監督4', 'シリーズ4', '制作会社D', '', 'ドラマ', 3500, '2025-09-03 05:16:48.412+00', '2025-09-03 05:16:48.412+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_004/api_video_004_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_004/api_video_004jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_004/api_video_004jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_004/api_video_004jp-3.jpg}', 'dmm', '2025-09-03 05:16:48.412+00', '2025-09-06 05:16:48.780354+00'),
	('b8016822-a058-4a4b-a300-e7efbd31baf8', 'api_video_005', 'API取得動画サンプル5', 'これはDMM API経由で取得された動画5です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_005/api_video_005pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_005/api_video_005_mhb_w.mp4', 'api_video_005', NULL, '監督5', 'シリーズ5', '制作会社E', '', 'コメディ', 4000, '2025-09-02 05:16:48.412+00', '2025-09-02 05:16:48.412+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_005/api_video_005_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_005/api_video_005jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_005/api_video_005jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_005/api_video_005jp-3.jpg}', 'dmm', '2025-09-02 05:16:48.412+00', '2025-09-06 05:16:48.83837+00'),
	('ff3a6103-c1b2-492a-8df4-370b72451d68', 'api_video_006', 'API取得動画サンプル6', 'これはDMM API経由で取得された動画6です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_006/api_video_006pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_006/api_video_006_mhb_w.mp4', 'api_video_006', NULL, '監督6', 'シリーズ6', '制作会社A', '', 'アクション', 4500, '2025-09-01 05:16:48.412+00', '2025-09-01 05:16:48.412+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_006/api_video_006_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_006/api_video_006jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_006/api_video_006jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_006/api_video_006jp-3.jpg}', 'dmm', '2025-09-01 05:16:48.412+00', '2025-09-06 05:16:48.907029+00'),
	('de3affb5-fd9c-4a4e-9033-a3341f7b695e', 'api_video_007', 'API取得動画サンプル7', 'これはDMM API経由で取得された動画7です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_007/api_video_007pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_007/api_video_007_mhb_w.mp4', 'api_video_007', NULL, '監督7', 'シリーズ7', '制作会社B', '', 'ドラマ', 5000, '2025-08-31 05:16:48.412+00', '2025-08-31 05:16:48.412+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_007/api_video_007_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_007/api_video_007jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_007/api_video_007jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_007/api_video_007jp-3.jpg}', 'dmm', '2025-08-31 05:16:48.412+00', '2025-09-06 05:16:48.964471+00'),
	('96e03fff-b705-409a-a686-c22cf8dc19d3', 'api_video_008', 'API取得動画サンプル8', 'これはDMM API経由で取得された動画8です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_008/api_video_008pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_008/api_video_008_mhb_w.mp4', 'api_video_008', NULL, '監督8', 'シリーズ8', '制作会社C', '', 'コメディ', 5500, '2025-08-30 05:16:48.412+00', '2025-08-30 05:16:48.412+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_008/api_video_008_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_008/api_video_008jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_008/api_video_008jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_008/api_video_008jp-3.jpg}', 'dmm', '2025-08-30 05:16:48.412+00', '2025-09-06 05:16:49.022566+00'),
	('6f442888-0fc6-4143-b85e-d5102ab63bfc', 'api_video_009', 'API取得動画サンプル9', 'これはDMM API経由で取得された動画9です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_009/api_video_009pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_009/api_video_009_mhb_w.mp4', 'api_video_009', NULL, '監督9', 'シリーズ9', '制作会社D', '', 'アクション', 6000, '2025-08-29 05:16:48.413+00', '2025-08-29 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_009/api_video_009_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_009/api_video_009jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_009/api_video_009jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_009/api_video_009jp-3.jpg}', 'dmm', '2025-08-29 05:16:48.413+00', '2025-09-06 05:16:49.083786+00'),
	('a6e83156-6fee-48b1-a3a6-809bc51eef8f', 'api_video_010', 'API取得動画サンプル10', 'これはDMM API経由で取得された動画10です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_010/api_video_010pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_010/api_video_010_mhb_w.mp4', 'api_video_010', NULL, '監督10', 'シリーズ10', '制作会社E', '', 'ドラマ', 6500, '2025-08-28 05:16:48.413+00', '2025-08-28 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_010/api_video_010_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_010/api_video_010jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_010/api_video_010jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_010/api_video_010jp-3.jpg}', 'dmm', '2025-08-28 05:16:48.413+00', '2025-09-06 05:16:49.134183+00'),
	('9878152d-c672-4dae-bbc2-4b8eaffbe184', 'api_video_011', 'API取得動画サンプル11', 'これはDMM API経由で取得された動画11です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_011/api_video_011pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_011/api_video_011_mhb_w.mp4', 'api_video_011', NULL, '監督11', 'シリーズ11', '制作会社A', '', 'コメディ', 7000, '2025-08-27 05:16:48.413+00', '2025-08-27 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_011/api_video_011_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_011/api_video_011jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_011/api_video_011jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_011/api_video_011jp-3.jpg}', 'dmm', '2025-08-27 05:16:48.413+00', '2025-09-06 05:16:49.162226+00'),
	('0ce31a5e-75e9-4fd8-bc57-bbe85f396d12', 'api_video_012', 'API取得動画サンプル12', 'これはDMM API経由で取得された動画12です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_012/api_video_012pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_012/api_video_012_mhb_w.mp4', 'api_video_012', NULL, '監督12', 'シリーズ12', '制作会社B', '', 'アクション', 7500, '2025-08-26 05:16:48.413+00', '2025-08-26 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_012/api_video_012_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_012/api_video_012jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_012/api_video_012jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_012/api_video_012jp-3.jpg}', 'dmm', '2025-08-26 05:16:48.413+00', '2025-09-06 05:16:49.188637+00'),
	('6ee24663-93a2-4a7d-8efb-eb099b5e2583', 'api_video_013', 'API取得動画サンプル13', 'これはDMM API経由で取得された動画13です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_013/api_video_013pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_013/api_video_013_mhb_w.mp4', 'api_video_013', NULL, '監督13', 'シリーズ13', '制作会社C', '', 'ドラマ', 8000, '2025-08-25 05:16:48.413+00', '2025-08-25 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_013/api_video_013_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_013/api_video_013jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_013/api_video_013jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_013/api_video_013jp-3.jpg}', 'dmm', '2025-08-25 05:16:48.413+00', '2025-09-06 05:16:49.212989+00'),
	('95ea2b98-3780-4d80-8090-407d828caa58', 'api_video_014', 'API取得動画サンプル14', 'これはDMM API経由で取得された動画14です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_014/api_video_014pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_014/api_video_014_mhb_w.mp4', 'api_video_014', NULL, '監督14', 'シリーズ14', '制作会社D', '', 'コメディ', 8500, '2025-08-24 05:16:48.413+00', '2025-08-24 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_014/api_video_014_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_014/api_video_014jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_014/api_video_014jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_014/api_video_014jp-3.jpg}', 'dmm', '2025-08-24 05:16:48.413+00', '2025-09-06 05:16:49.238806+00'),
	('9f8ddd96-81be-4619-a7e4-6d6a1f852155', 'api_video_015', 'API取得動画サンプル15', 'これはDMM API経由で取得された動画15です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_015/api_video_015pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_015/api_video_015_mhb_w.mp4', 'api_video_015', NULL, '監督15', 'シリーズ15', '制作会社E', '', 'アクション', 9000, '2025-08-23 05:16:48.413+00', '2025-08-23 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_015/api_video_015_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_015/api_video_015jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_015/api_video_015jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_015/api_video_015jp-3.jpg}', 'dmm', '2025-08-23 05:16:48.413+00', '2025-09-06 05:16:49.262748+00'),
	('e51f1cfd-1e90-4aed-bf33-8c292f8d2003', 'api_video_016', 'API取得動画サンプル16', 'これはDMM API経由で取得された動画16です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_016/api_video_016pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_016/api_video_016_mhb_w.mp4', 'api_video_016', NULL, '監督16', 'シリーズ16', '制作会社A', '', 'ドラマ', 9500, '2025-08-22 05:16:48.413+00', '2025-08-22 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_016/api_video_016_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_016/api_video_016jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_016/api_video_016jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_016/api_video_016jp-3.jpg}', 'dmm', '2025-08-22 05:16:48.413+00', '2025-09-06 05:16:49.288701+00'),
	('8aeb0571-e412-4846-b3e0-f99bb9d524a1', 'api_video_017', 'API取得動画サンプル17', 'これはDMM API経由で取得された動画17です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_017/api_video_017pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_017/api_video_017_mhb_w.mp4', 'api_video_017', NULL, '監督17', 'シリーズ17', '制作会社B', '', 'コメディ', 10000, '2025-08-21 05:16:48.413+00', '2025-08-21 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_017/api_video_017_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_017/api_video_017jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_017/api_video_017jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_017/api_video_017jp-3.jpg}', 'dmm', '2025-08-21 05:16:48.413+00', '2025-09-06 05:16:49.313168+00'),
	('afea974d-fe61-4b59-b62c-e0db048fc05c', 'api_video_018', 'API取得動画サンプル18', 'これはDMM API経由で取得された動画18です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_018/api_video_018pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_018/api_video_018_mhb_w.mp4', 'api_video_018', NULL, '監督18', 'シリーズ18', '制作会社C', '', 'アクション', 10500, '2025-08-20 05:16:48.413+00', '2025-08-20 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_018/api_video_018_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_018/api_video_018jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_018/api_video_018jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_018/api_video_018jp-3.jpg}', 'dmm', '2025-08-20 05:16:48.413+00', '2025-09-06 05:16:49.358257+00'),
	('e476db97-414f-454d-a3ed-6816c86fd595', 'api_video_019', 'API取得動画サンプル19', 'これはDMM API経由で取得された動画19です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_019/api_video_019pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_019/api_video_019_mhb_w.mp4', 'api_video_019', NULL, '監督19', 'シリーズ19', '制作会社D', '', 'ドラマ', 11000, '2025-08-19 05:16:48.413+00', '2025-08-19 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_019/api_video_019_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_019/api_video_019jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_019/api_video_019jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_019/api_video_019jp-3.jpg}', 'dmm', '2025-08-19 05:16:48.413+00', '2025-09-06 05:16:49.384127+00'),
	('d48e46e7-d92f-4a74-85ac-387dad069899', 'api_video_020', 'API取得動画サンプル20', 'これはDMM API経由で取得された動画20です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_020/api_video_020pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_020/api_video_020_mhb_w.mp4', 'api_video_020', NULL, '監督20', 'シリーズ20', '制作会社E', '', 'コメディ', 11500, '2025-08-18 05:16:48.413+00', '2025-08-18 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_020/api_video_020_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_020/api_video_020jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_020/api_video_020jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_020/api_video_020jp-3.jpg}', 'dmm', '2025-08-18 05:16:48.413+00', '2025-09-06 05:16:49.435041+00'),
	('44f5c2f6-d4c4-4d98-aee5-1040d9ff9ea5', 'api_video_021', 'API取得動画サンプル21', 'これはDMM API経由で取得された動画21です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_021/api_video_021pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_021/api_video_021_mhb_w.mp4', 'api_video_021', NULL, '監督21', 'シリーズ21', '制作会社A', '', 'アクション', 12000, '2025-08-17 05:16:48.413+00', '2025-08-17 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_021/api_video_021_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_021/api_video_021jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_021/api_video_021jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_021/api_video_021jp-3.jpg}', 'dmm', '2025-08-17 05:16:48.413+00', '2025-09-06 05:16:49.479667+00'),
	('f1221582-31be-43fc-a83a-f85e1631d0e3', 'api_video_022', 'API取得動画サンプル22', 'これはDMM API経由で取得された動画22です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_022/api_video_022pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_022/api_video_022_mhb_w.mp4', 'api_video_022', NULL, '監督22', 'シリーズ22', '制作会社B', '', 'ドラマ', 12500, '2025-08-16 05:16:48.413+00', '2025-08-16 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_022/api_video_022_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_022/api_video_022jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_022/api_video_022jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_022/api_video_022jp-3.jpg}', 'dmm', '2025-08-16 05:16:48.413+00', '2025-09-06 05:16:49.531975+00'),
	('e5b2f71d-20d2-48ab-a344-63e3553cddcc', 'api_video_023', 'API取得動画サンプル23', 'これはDMM API経由で取得された動画23です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_023/api_video_023pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_023/api_video_023_mhb_w.mp4', 'api_video_023', NULL, '監督23', 'シリーズ23', '制作会社C', '', 'コメディ', 13000, '2025-08-15 05:16:48.413+00', '2025-08-15 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_023/api_video_023_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_023/api_video_023jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_023/api_video_023jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_023/api_video_023jp-3.jpg}', 'dmm', '2025-08-15 05:16:48.413+00', '2025-09-06 05:16:49.596824+00'),
	('f0c41f92-5b75-489f-a178-34a565baac39', 'api_video_024', 'API取得動画サンプル24', 'これはDMM API経由で取得された動画24です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_024/api_video_024pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_024/api_video_024_mhb_w.mp4', 'api_video_024', NULL, '監督24', 'シリーズ24', '制作会社D', '', 'アクション', 13500, '2025-08-14 05:16:48.413+00', '2025-08-14 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_024/api_video_024_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_024/api_video_024jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_024/api_video_024jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_024/api_video_024jp-3.jpg}', 'dmm', '2025-08-14 05:16:48.413+00', '2025-09-06 05:16:49.650067+00'),
	('890f36f1-5382-4b3a-98bc-2a121d02859c', 'api_video_025', 'API取得動画サンプル25', 'これはDMM API経由で取得された動画25です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_025/api_video_025pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_025/api_video_025_mhb_w.mp4', 'api_video_025', NULL, '監督25', 'シリーズ25', '制作会社E', '', 'ドラマ', 14000, '2025-08-13 05:16:48.413+00', '2025-08-13 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_025/api_video_025_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_025/api_video_025jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_025/api_video_025jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_025/api_video_025jp-3.jpg}', 'dmm', '2025-08-13 05:16:48.413+00', '2025-09-06 05:16:49.708141+00'),
	('a76f581e-1dcf-451b-8776-4eb3b22d58a8', 'api_video_026', 'API取得動画サンプル26', 'これはDMM API経由で取得された動画26です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_026/api_video_026pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_026/api_video_026_mhb_w.mp4', 'api_video_026', NULL, '監督26', 'シリーズ26', '制作会社A', '', 'コメディ', 14500, '2025-08-12 05:16:48.413+00', '2025-08-12 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_026/api_video_026_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_026/api_video_026jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_026/api_video_026jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_026/api_video_026jp-3.jpg}', 'dmm', '2025-08-12 05:16:48.413+00', '2025-09-06 05:16:49.765842+00'),
	('419a6dc8-c92d-49d5-93a7-5d10f17a0329', 'api_video_027', 'API取得動画サンプル27', 'これはDMM API経由で取得された動画27です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_027/api_video_027pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_027/api_video_027_mhb_w.mp4', 'api_video_027', NULL, '監督27', 'シリーズ27', '制作会社B', '', 'アクション', 15000, '2025-08-11 05:16:48.413+00', '2025-08-11 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_027/api_video_027_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_027/api_video_027jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_027/api_video_027jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_027/api_video_027jp-3.jpg}', 'dmm', '2025-08-11 05:16:48.413+00', '2025-09-06 05:16:49.818044+00'),
	('da2dbbb4-21f6-4a3a-b581-71f971cc454f', 'api_video_028', 'API取得動画サンプル28', 'これはDMM API経由で取得された動画28です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_028/api_video_028pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_028/api_video_028_mhb_w.mp4', 'api_video_028', NULL, '監督28', 'シリーズ28', '制作会社C', '', 'ドラマ', 15500, '2025-08-10 05:16:48.413+00', '2025-08-10 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_028/api_video_028_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_028/api_video_028jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_028/api_video_028jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_028/api_video_028jp-3.jpg}', 'dmm', '2025-08-10 05:16:48.413+00', '2025-09-06 05:16:49.868263+00'),
	('a140b277-9369-4a9e-a0d7-04a5d2000f91', 'api_video_029', 'API取得動画サンプル29', 'これはDMM API経由で取得された動画29です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_029/api_video_029pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_029/api_video_029_mhb_w.mp4', 'api_video_029', NULL, '監督29', 'シリーズ29', '制作会社D', '', 'コメディ', 16000, '2025-08-09 05:16:48.413+00', '2025-08-09 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_029/api_video_029_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_029/api_video_029jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_029/api_video_029jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_029/api_video_029jp-3.jpg}', 'dmm', '2025-08-09 05:16:48.413+00', '2025-09-06 05:16:49.922121+00'),
	('530014c0-8128-4eb2-879b-ef1e2becc4df', 'api_video_030', 'API取得動画サンプル30', 'これはDMM API経由で取得された動画30です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_030/api_video_030pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_030/api_video_030_mhb_w.mp4', 'api_video_030', NULL, '監督30', 'シリーズ30', '制作会社E', '', 'アクション', 16500, '2025-08-08 05:16:48.413+00', '2025-08-08 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_030/api_video_030_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_030/api_video_030jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_030/api_video_030jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_030/api_video_030jp-3.jpg}', 'dmm', '2025-08-08 05:16:48.413+00', '2025-09-06 05:16:49.978752+00'),
	('d8656d7d-f56f-452b-ae07-d1f89facf61b', 'api_video_031', 'API取得動画サンプル31', 'これはDMM API経由で取得された動画31です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_031/api_video_031pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_031/api_video_031_mhb_w.mp4', 'api_video_031', NULL, '監督31', 'シリーズ31', '制作会社A', '', 'ドラマ', 17000, '2025-08-07 05:16:48.413+00', '2025-08-07 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_031/api_video_031_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_031/api_video_031jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_031/api_video_031jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_031/api_video_031jp-3.jpg}', 'dmm', '2025-08-07 05:16:48.413+00', '2025-09-06 05:16:50.033618+00'),
	('8c42f302-08cc-41fa-ac1b-032661be1a12', 'api_video_032', 'API取得動画サンプル32', 'これはDMM API経由で取得された動画32です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_032/api_video_032pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_032/api_video_032_mhb_w.mp4', 'api_video_032', NULL, '監督32', 'シリーズ32', '制作会社B', '', 'コメディ', 17500, '2025-08-06 05:16:48.413+00', '2025-08-06 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_032/api_video_032_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_032/api_video_032jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_032/api_video_032jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_032/api_video_032jp-3.jpg}', 'dmm', '2025-08-06 05:16:48.413+00', '2025-09-06 05:16:50.09053+00'),
	('e269f26e-a036-4b95-afa2-7246b70ddead', 'api_video_033', 'API取得動画サンプル33', 'これはDMM API経由で取得された動画33です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_033/api_video_033pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_033/api_video_033_mhb_w.mp4', 'api_video_033', NULL, '監督33', 'シリーズ33', '制作会社C', '', 'アクション', 18000, '2025-08-05 05:16:48.413+00', '2025-08-05 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_033/api_video_033_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_033/api_video_033jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_033/api_video_033jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_033/api_video_033jp-3.jpg}', 'dmm', '2025-08-05 05:16:48.413+00', '2025-09-06 05:16:50.162659+00'),
	('7c6e67f4-916d-46d7-b2f6-3ded7779e4ec', 'api_video_034', 'API取得動画サンプル34', 'これはDMM API経由で取得された動画34です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_034/api_video_034pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_034/api_video_034_mhb_w.mp4', 'api_video_034', NULL, '監督34', 'シリーズ34', '制作会社D', '', 'ドラマ', 18500, '2025-08-04 05:16:48.413+00', '2025-08-04 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_034/api_video_034_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_034/api_video_034jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_034/api_video_034jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_034/api_video_034jp-3.jpg}', 'dmm', '2025-08-04 05:16:48.413+00', '2025-09-06 05:16:50.23701+00'),
	('1848141f-2a61-46b3-bda1-e932f9fcbd57', 'api_video_035', 'API取得動画サンプル35', 'これはDMM API経由で取得された動画35です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_035/api_video_035pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_035/api_video_035_mhb_w.mp4', 'api_video_035', NULL, '監督35', 'シリーズ35', '制作会社E', '', 'コメディ', 19000, '2025-08-03 05:16:48.413+00', '2025-08-03 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_035/api_video_035_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_035/api_video_035jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_035/api_video_035jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_035/api_video_035jp-3.jpg}', 'dmm', '2025-08-03 05:16:48.413+00', '2025-09-06 05:16:50.306971+00'),
	('1eaaf1a4-5479-4385-8e6f-5facea6e1e79', 'api_video_036', 'API取得動画サンプル36', 'これはDMM API経由で取得された動画36です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_036/api_video_036pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_036/api_video_036_mhb_w.mp4', 'api_video_036', NULL, '監督36', 'シリーズ36', '制作会社A', '', 'アクション', 19500, '2025-08-02 05:16:48.413+00', '2025-08-02 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_036/api_video_036_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_036/api_video_036jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_036/api_video_036jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_036/api_video_036jp-3.jpg}', 'dmm', '2025-08-02 05:16:48.413+00', '2025-09-06 05:16:50.372742+00'),
	('a1dbd6f0-2297-4222-8f05-f3af380334b3', 'api_video_037', 'API取得動画サンプル37', 'これはDMM API経由で取得された動画37です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_037/api_video_037pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_037/api_video_037_mhb_w.mp4', 'api_video_037', NULL, '監督37', 'シリーズ37', '制作会社B', '', 'ドラマ', 20000, '2025-08-01 05:16:48.413+00', '2025-08-01 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_037/api_video_037_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_037/api_video_037jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_037/api_video_037jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_037/api_video_037jp-3.jpg}', 'dmm', '2025-08-01 05:16:48.413+00', '2025-09-06 05:16:50.425124+00'),
	('b5c236c9-18ec-4fb9-800f-d2befc69b770', 'api_video_038', 'API取得動画サンプル38', 'これはDMM API経由で取得された動画38です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_038/api_video_038pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_038/api_video_038_mhb_w.mp4', 'api_video_038', NULL, '監督38', 'シリーズ38', '制作会社C', '', 'コメディ', 20500, '2025-07-31 05:16:48.413+00', '2025-07-31 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_038/api_video_038_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_038/api_video_038jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_038/api_video_038jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_038/api_video_038jp-3.jpg}', 'dmm', '2025-07-31 05:16:48.413+00', '2025-09-06 05:16:50.461032+00'),
	('9d549281-4049-4aa5-9f6f-3e0f948286c4', 'api_video_039', 'API取得動画サンプル39', 'これはDMM API経由で取得された動画39です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_039/api_video_039pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_039/api_video_039_mhb_w.mp4', 'api_video_039', NULL, '監督39', 'シリーズ39', '制作会社D', '', 'アクション', 21000, '2025-07-30 05:16:48.413+00', '2025-07-30 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_039/api_video_039_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_039/api_video_039jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_039/api_video_039jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_039/api_video_039jp-3.jpg}', 'dmm', '2025-07-30 05:16:48.413+00', '2025-09-06 05:16:50.497417+00'),
	('91ed035e-19c7-44d7-9732-b1ada432c9b8', 'api_video_040', 'API取得動画サンプル40', 'これはDMM API経由で取得された動画40です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_040/api_video_040pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_040/api_video_040_mhb_w.mp4', 'api_video_040', NULL, '監督40', 'シリーズ40', '制作会社E', '', 'ドラマ', 21500, '2025-07-29 05:16:48.413+00', '2025-07-29 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_040/api_video_040_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_040/api_video_040jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_040/api_video_040jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_040/api_video_040jp-3.jpg}', 'dmm', '2025-07-29 05:16:48.413+00', '2025-09-06 05:16:50.542362+00'),
	('1949a2bf-336c-483c-ab71-bc7a6ba688d9', 'api_video_041', 'API取得動画サンプル41', 'これはDMM API経由で取得された動画41です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_041/api_video_041pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_041/api_video_041_mhb_w.mp4', 'api_video_041', NULL, '監督41', 'シリーズ41', '制作会社A', '', 'コメディ', 22000, '2025-07-28 05:16:48.413+00', '2025-07-28 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_041/api_video_041_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_041/api_video_041jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_041/api_video_041jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_041/api_video_041jp-3.jpg}', 'dmm', '2025-07-28 05:16:48.413+00', '2025-09-06 05:16:50.577707+00'),
	('d676eaad-e18c-40cd-9700-f657e8b5ba6f', 'api_video_042', 'API取得動画サンプル42', 'これはDMM API経由で取得された動画42です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_042/api_video_042pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_042/api_video_042_mhb_w.mp4', 'api_video_042', NULL, '監督42', 'シリーズ42', '制作会社B', '', 'アクション', 22500, '2025-07-27 05:16:48.413+00', '2025-07-27 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_042/api_video_042_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_042/api_video_042jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_042/api_video_042jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_042/api_video_042jp-3.jpg}', 'dmm', '2025-07-27 05:16:48.413+00', '2025-09-06 05:16:50.609275+00'),
	('bd14a621-fb73-4328-93a5-7c7dce6f6785', 'api_video_043', 'API取得動画サンプル43', 'これはDMM API経由で取得された動画43です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_043/api_video_043pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_043/api_video_043_mhb_w.mp4', 'api_video_043', NULL, '監督43', 'シリーズ43', '制作会社C', '', 'ドラマ', 23000, '2025-07-26 05:16:48.413+00', '2025-07-26 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_043/api_video_043_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_043/api_video_043jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_043/api_video_043jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_043/api_video_043jp-3.jpg}', 'dmm', '2025-07-26 05:16:48.413+00', '2025-09-06 05:16:50.638916+00'),
	('842b8e45-a7a0-43fd-a97f-bbc08729ba64', 'api_video_044', 'API取得動画サンプル44', 'これはDMM API経由で取得された動画44です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_044/api_video_044pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_044/api_video_044_mhb_w.mp4', 'api_video_044', NULL, '監督44', 'シリーズ44', '制作会社D', '', 'コメディ', 23500, '2025-07-25 05:16:48.413+00', '2025-07-25 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_044/api_video_044_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_044/api_video_044jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_044/api_video_044jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_044/api_video_044jp-3.jpg}', 'dmm', '2025-07-25 05:16:48.413+00', '2025-09-06 05:16:50.669733+00'),
	('639e1ea4-7406-4a09-a89c-cf5f0e8e260b', 'api_video_045', 'API取得動画サンプル45', 'これはDMM API経由で取得された動画45です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_045/api_video_045pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_045/api_video_045_mhb_w.mp4', 'api_video_045', NULL, '監督45', 'シリーズ45', '制作会社E', '', 'アクション', 24000, '2025-07-24 05:16:48.413+00', '2025-07-24 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_045/api_video_045_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_045/api_video_045jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_045/api_video_045jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_045/api_video_045jp-3.jpg}', 'dmm', '2025-07-24 05:16:48.413+00', '2025-09-06 05:16:50.703252+00'),
	('26e4101b-c70b-4907-b6ac-9b25aa62d1df', 'api_video_046', 'API取得動画サンプル46', 'これはDMM API経由で取得された動画46です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_046/api_video_046pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_046/api_video_046_mhb_w.mp4', 'api_video_046', NULL, '監督46', 'シリーズ46', '制作会社A', '', 'ドラマ', 24500, '2025-07-23 05:16:48.413+00', '2025-07-23 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_046/api_video_046_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_046/api_video_046jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_046/api_video_046jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_046/api_video_046jp-3.jpg}', 'dmm', '2025-07-23 05:16:48.413+00', '2025-09-06 05:16:50.732022+00'),
	('5a1865fe-5ec0-4667-bb28-5f21bd38b095', 'api_video_047', 'API取得動画サンプル47', 'これはDMM API経由で取得された動画47です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_047/api_video_047pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_047/api_video_047_mhb_w.mp4', 'api_video_047', NULL, '監督47', 'シリーズ47', '制作会社B', '', 'コメディ', 25000, '2025-07-22 05:16:48.413+00', '2025-07-22 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_047/api_video_047_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_047/api_video_047jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_047/api_video_047jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_047/api_video_047jp-3.jpg}', 'dmm', '2025-07-22 05:16:48.413+00', '2025-09-06 05:16:50.76158+00'),
	('86ed453d-766d-4876-a5be-e621786e0e3a', 'api_video_048', 'API取得動画サンプル48', 'これはDMM API経由で取得された動画48です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_048/api_video_048pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_048/api_video_048_mhb_w.mp4', 'api_video_048', NULL, '監督48', 'シリーズ48', '制作会社C', '', 'アクション', 25500, '2025-07-21 05:16:48.413+00', '2025-07-21 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_048/api_video_048_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_048/api_video_048jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_048/api_video_048jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_048/api_video_048jp-3.jpg}', 'dmm', '2025-07-21 05:16:48.413+00', '2025-09-06 05:16:50.790718+00'),
	('a5d7b33c-714b-43ae-ab67-8195d154fabc', 'api_video_049', 'API取得動画サンプル49', 'これはDMM API経由で取得された動画49です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_049/api_video_049pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_049/api_video_049_mhb_w.mp4', 'api_video_049', NULL, '監督49', 'シリーズ49', '制作会社D', '', 'ドラマ', 26000, '2025-07-20 05:16:48.413+00', '2025-07-20 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_049/api_video_049_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_049/api_video_049jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_049/api_video_049jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_049/api_video_049jp-3.jpg}', 'dmm', '2025-07-20 05:16:48.413+00', '2025-09-06 05:16:50.823604+00'),
	('a9d8280d-9752-491b-a716-086d390528d8', 'api_video_050', 'API取得動画サンプル50', 'これはDMM API経由で取得された動画50です', NULL, 'https://pics.dmm.co.jp/digital/video/api_video_050/api_video_050pl.jpg', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_050/api_video_050_mhb_w.mp4', 'api_video_050', NULL, '監督50', 'シリーズ50', '制作会社E', '', 'コメディ', 26500, '2025-07-19 05:16:48.413+00', '2025-07-19 05:16:48.413+00', 'https://cc3001.dmm.co.jp/litevideo/freepv/api_video_050/api_video_050_mhb_w.mp4', '{https://pics.dmm.co.jp/digital/video/api_video_050/api_video_050jp-1.jpg,https://pics.dmm.co.jp/digital/video/api_video_050/api_video_050jp-2.jpg,https://pics.dmm.co.jp/digital/video/api_video_050/api_video_050jp-3.jpg}', 'dmm', '2025-07-19 05:16:48.413+00', '2025-09-06 05:16:50.853431+00');


--
-- Data for Name: likes; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO "public"."likes" ("user_id", "video_id", "purchased", "created_at") VALUES
	('e010209e-1e56-450d-bf5f-76a478a9c6ca', '550e8400-e29b-41d4-a716-446655441000', false, '2025-09-01 10:00:00+00'),
	('e010209e-1e56-450d-bf5f-76a478a9c6ca', '550e8400-e29b-41d4-a716-446655441001', false, '2025-09-01 11:00:00+00');


--
-- Data for Name: performers; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO "public"."performers" ("id", "name") VALUES
	('550e8400-e29b-41d4-a716-446655440100', '田中美咲'),
	('550e8400-e29b-41d4-a716-446655440101', '佐藤花音'),
	('550e8400-e29b-41d4-a716-446655440102', '鈴木あい'),
	('55ee0819-591d-4f1c-ace5-db23b2b45eaf', 'テスト出演者1'),
	('1480734b-f2b7-4266-b724-61267603f7d0', '出演者1'),
	('48ce503e-d0f3-43ec-8a92-984629b0a66a', '出演者2'),
	('bbec2df7-90ee-468c-a11a-3477e24b4c43', '出演者3'),
	('2534b204-1dfa-499a-9d5a-78d0960594ef', '出演者4'),
	('0f7c85f0-7cee-44c2-8121-633b985fdb23', '出演者5'),
	('fe890eda-30be-4280-9698-f5b63c387cfe', '出演者6'),
	('4e00b01b-d2ac-47a8-9470-aa7e7ca67601', '出演者7'),
	('c58cf823-4dd5-4332-b67d-cb9534a40615', '出演者8'),
	('8de4d3a2-11fe-414b-818b-05ab8f48d41f', '出演者9'),
	('7104357a-abbd-4b29-bc4e-5dc093e1ca9e', '出演者10'),
	('2377369b-9589-49e0-8f0b-c720a175a774', '出演者11'),
	('7ac5e02f-4506-4c4e-8869-07808135d75e', '出演者12'),
	('4dd50395-f4f2-4044-8470-fe8a7229a581', '出演者13'),
	('5af02306-ca38-469d-ad8e-8f326cb632c2', '出演者14'),
	('50836674-c724-4763-9fa3-f8c831ebca4d', '出演者15'),
	('3bac2cdc-63d2-45f5-9f7c-4d373199312d', '出演者16'),
	('1ad8b01e-3637-472f-a6eb-b903902e7d49', '出演者17'),
	('ec579d50-68e6-4b30-be9e-100213de203d', '出演者18'),
	('1f5c1025-020a-4e50-9dd7-ce2dfa4aef4b', '出演者19'),
	('3479dc26-e042-4e7c-907d-396644cec6b2', '出演者20'),
	('92e05741-c464-47e8-b7f6-9bd84dd919eb', '出演者21'),
	('79b9f77f-0a69-4038-9a5e-e5d776397c9c', '出演者22'),
	('4524bcce-753a-4d4d-a226-977dc911a289', '出演者23'),
	('23b0f0dd-6b8e-477a-990f-fc17850a4778', '出演者24'),
	('e026b21c-776c-4726-acdd-73e1992525c5', '出演者25'),
	('a5f8fa7d-c191-4379-81f9-4b894b478e36', '出演者26'),
	('51cd7b8d-fe7c-48e0-a560-7d079340654d', '出演者27'),
	('ceac6e7d-a675-49f3-a875-6753ced1f0be', '出演者28'),
	('be6e89bd-380a-49e2-850b-80439f872507', '出演者29'),
	('80086075-df3e-4b86-af63-35dc23c59e34', '出演者30'),
	('b7dd6517-c1f6-43a1-abca-38a3f0cfbcc7', '出演者31'),
	('777ffcc1-049e-4a65-b593-0e685d050b0c', '出演者32'),
	('1880bf07-0b4a-4029-a9d7-93a2a14505a0', '出演者33'),
	('b0c4e460-5950-4bc2-b118-e1b645a65bd9', '出演者34'),
	('456db0ff-66c0-46d1-9fb8-214ec87568d3', '出演者35'),
	('3ddb6ce3-ae34-4893-8a23-cb42aff5d7ae', '出演者36'),
	('5f8fad3c-f5e6-4027-b472-aabc922f1b90', '出演者37'),
	('f16cc6e9-741c-4e5a-a303-1c59220adcf2', '出演者38'),
	('92529601-dbf1-4227-937d-363c555aeaab', '出演者39'),
	('962c9341-3d9b-4191-8591-4e96fdbfbec2', '出演者40'),
	('cee660b8-91b2-4343-bead-69eb5f689f8e', '出演者41'),
	('8e075f68-a534-45dc-b2b0-a3e9516a61b1', '出演者42'),
	('1cb7d563-5741-4dc2-99d0-cd28d7fb60e4', '出演者43'),
	('692c75e8-28dc-41e2-ad56-447792fd2db8', '出演者44'),
	('7639b5df-7f88-467c-afbe-184a6362ad72', '出演者45'),
	('0be3a46b-46d3-47b1-9d37-68a1b2028de4', '出演者46'),
	('34709f58-9bfc-45ec-a642-a4693b9ad1ea', '出演者47'),
	('75b9701e-56b8-4807-a98b-052100fd328f', '出演者48'),
	('8d022075-4414-4ab4-a7ef-5cea4f1d0662', '出演者49'),
	('3960da9e-c1d8-4282-a9f4-a1660b7ff07e', '出演者50');


--
-- Data for Name: profiles; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- Data for Name: tag_groups; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO "public"."tag_groups" ("id", "name") VALUES
	('550e8400-e29b-41d4-a716-446655440001', 'ジャンル'),
	('550e8400-e29b-41d4-a716-446655440002', 'プレイ');


--
-- Data for Name: tags; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO "public"."tags" ("id", "name", "tag_group_id") VALUES
	('550e8400-e29b-41d4-a716-446655440010', 'ドラマ', '550e8400-e29b-41d4-a716-446655440001'),
	('550e8400-e29b-41d4-a716-446655440011', 'コメディ', '550e8400-e29b-41d4-a716-446655440001'),
	('550e8400-e29b-41d4-a716-446655440012', 'アクション', '550e8400-e29b-41d4-a716-446655440001'),
	('550e8400-e29b-41d4-a716-446655440020', 'ソフト', '550e8400-e29b-41d4-a716-446655440002'),
	('550e8400-e29b-41d4-a716-446655440021', 'ハード', '550e8400-e29b-41d4-a716-446655440002'),
	('4b6b4ca1-5e38-4506-ae70-a1947e1c42ca', 'テストジャンル', '550e8400-e29b-41d4-a716-446655440001');


--
-- Data for Name: user_embeddings; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- Data for Name: video_embeddings; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- Data for Name: video_performers; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO "public"."video_performers" ("video_id", "performer_id") VALUES
	('550e8400-e29b-41d4-a716-446655441000', '550e8400-e29b-41d4-a716-446655440100'),
	('550e8400-e29b-41d4-a716-446655441001', '550e8400-e29b-41d4-a716-446655440101'),
	('550e8400-e29b-41d4-a716-446655441002', '550e8400-e29b-41d4-a716-446655440102'),
	('550e8400-e29b-41d4-a716-446655441003', '550e8400-e29b-41d4-a716-446655440100'),
	('550e8400-e29b-41d4-a716-446655441004', '550e8400-e29b-41d4-a716-446655440101'),
	('87fd544b-024c-441a-828d-4cd920ecd138', '55ee0819-591d-4f1c-ace5-db23b2b45eaf'),
	('0aa57ba7-b627-45b4-b01d-85a4c4ce0e2b', '1480734b-f2b7-4266-b724-61267603f7d0'),
	('21a62476-e985-4cbd-b976-8f50d78c0b50', '48ce503e-d0f3-43ec-8a92-984629b0a66a'),
	('b4d6190b-4f08-410a-8759-15fd7d732b4b', 'bbec2df7-90ee-468c-a11a-3477e24b4c43'),
	('740ca061-4a04-452c-9a8c-0037363125c8', '2534b204-1dfa-499a-9d5a-78d0960594ef'),
	('b8016822-a058-4a4b-a300-e7efbd31baf8', '0f7c85f0-7cee-44c2-8121-633b985fdb23'),
	('ff3a6103-c1b2-492a-8df4-370b72451d68', 'fe890eda-30be-4280-9698-f5b63c387cfe'),
	('de3affb5-fd9c-4a4e-9033-a3341f7b695e', '4e00b01b-d2ac-47a8-9470-aa7e7ca67601'),
	('96e03fff-b705-409a-a686-c22cf8dc19d3', 'c58cf823-4dd5-4332-b67d-cb9534a40615'),
	('6f442888-0fc6-4143-b85e-d5102ab63bfc', '8de4d3a2-11fe-414b-818b-05ab8f48d41f'),
	('a6e83156-6fee-48b1-a3a6-809bc51eef8f', '7104357a-abbd-4b29-bc4e-5dc093e1ca9e'),
	('9878152d-c672-4dae-bbc2-4b8eaffbe184', '2377369b-9589-49e0-8f0b-c720a175a774'),
	('0ce31a5e-75e9-4fd8-bc57-bbe85f396d12', '7ac5e02f-4506-4c4e-8869-07808135d75e'),
	('6ee24663-93a2-4a7d-8efb-eb099b5e2583', '4dd50395-f4f2-4044-8470-fe8a7229a581'),
	('95ea2b98-3780-4d80-8090-407d828caa58', '5af02306-ca38-469d-ad8e-8f326cb632c2'),
	('9f8ddd96-81be-4619-a7e4-6d6a1f852155', '50836674-c724-4763-9fa3-f8c831ebca4d'),
	('e51f1cfd-1e90-4aed-bf33-8c292f8d2003', '3bac2cdc-63d2-45f5-9f7c-4d373199312d'),
	('8aeb0571-e412-4846-b3e0-f99bb9d524a1', '1ad8b01e-3637-472f-a6eb-b903902e7d49'),
	('afea974d-fe61-4b59-b62c-e0db048fc05c', 'ec579d50-68e6-4b30-be9e-100213de203d'),
	('e476db97-414f-454d-a3ed-6816c86fd595', '1f5c1025-020a-4e50-9dd7-ce2dfa4aef4b'),
	('d48e46e7-d92f-4a74-85ac-387dad069899', '3479dc26-e042-4e7c-907d-396644cec6b2'),
	('44f5c2f6-d4c4-4d98-aee5-1040d9ff9ea5', '92e05741-c464-47e8-b7f6-9bd84dd919eb'),
	('f1221582-31be-43fc-a83a-f85e1631d0e3', '79b9f77f-0a69-4038-9a5e-e5d776397c9c'),
	('e5b2f71d-20d2-48ab-a344-63e3553cddcc', '4524bcce-753a-4d4d-a226-977dc911a289'),
	('f0c41f92-5b75-489f-a178-34a565baac39', '23b0f0dd-6b8e-477a-990f-fc17850a4778'),
	('890f36f1-5382-4b3a-98bc-2a121d02859c', 'e026b21c-776c-4726-acdd-73e1992525c5'),
	('a76f581e-1dcf-451b-8776-4eb3b22d58a8', 'a5f8fa7d-c191-4379-81f9-4b894b478e36'),
	('419a6dc8-c92d-49d5-93a7-5d10f17a0329', '51cd7b8d-fe7c-48e0-a560-7d079340654d'),
	('da2dbbb4-21f6-4a3a-b581-71f971cc454f', 'ceac6e7d-a675-49f3-a875-6753ced1f0be'),
	('a140b277-9369-4a9e-a0d7-04a5d2000f91', 'be6e89bd-380a-49e2-850b-80439f872507'),
	('530014c0-8128-4eb2-879b-ef1e2becc4df', '80086075-df3e-4b86-af63-35dc23c59e34'),
	('d8656d7d-f56f-452b-ae07-d1f89facf61b', 'b7dd6517-c1f6-43a1-abca-38a3f0cfbcc7'),
	('8c42f302-08cc-41fa-ac1b-032661be1a12', '777ffcc1-049e-4a65-b593-0e685d050b0c'),
	('e269f26e-a036-4b95-afa2-7246b70ddead', '1880bf07-0b4a-4029-a9d7-93a2a14505a0'),
	('7c6e67f4-916d-46d7-b2f6-3ded7779e4ec', 'b0c4e460-5950-4bc2-b118-e1b645a65bd9'),
	('1848141f-2a61-46b3-bda1-e932f9fcbd57', '456db0ff-66c0-46d1-9fb8-214ec87568d3'),
	('1eaaf1a4-5479-4385-8e6f-5facea6e1e79', '3ddb6ce3-ae34-4893-8a23-cb42aff5d7ae'),
	('a1dbd6f0-2297-4222-8f05-f3af380334b3', '5f8fad3c-f5e6-4027-b472-aabc922f1b90'),
	('b5c236c9-18ec-4fb9-800f-d2befc69b770', 'f16cc6e9-741c-4e5a-a303-1c59220adcf2'),
	('9d549281-4049-4aa5-9f6f-3e0f948286c4', '92529601-dbf1-4227-937d-363c555aeaab'),
	('91ed035e-19c7-44d7-9732-b1ada432c9b8', '962c9341-3d9b-4191-8591-4e96fdbfbec2'),
	('1949a2bf-336c-483c-ab71-bc7a6ba688d9', 'cee660b8-91b2-4343-bead-69eb5f689f8e'),
	('d676eaad-e18c-40cd-9700-f657e8b5ba6f', '8e075f68-a534-45dc-b2b0-a3e9516a61b1'),
	('bd14a621-fb73-4328-93a5-7c7dce6f6785', '1cb7d563-5741-4dc2-99d0-cd28d7fb60e4'),
	('842b8e45-a7a0-43fd-a97f-bbc08729ba64', '692c75e8-28dc-41e2-ad56-447792fd2db8'),
	('639e1ea4-7406-4a09-a89c-cf5f0e8e260b', '7639b5df-7f88-467c-afbe-184a6362ad72'),
	('26e4101b-c70b-4907-b6ac-9b25aa62d1df', '0be3a46b-46d3-47b1-9d37-68a1b2028de4'),
	('5a1865fe-5ec0-4667-bb28-5f21bd38b095', '34709f58-9bfc-45ec-a642-a4693b9ad1ea'),
	('86ed453d-766d-4876-a5be-e621786e0e3a', '75b9701e-56b8-4807-a98b-052100fd328f'),
	('a5d7b33c-714b-43ae-ab67-8195d154fabc', '8d022075-4414-4ab4-a7ef-5cea4f1d0662'),
	('a9d8280d-9752-491b-a716-086d390528d8', '3960da9e-c1d8-4282-a9f4-a1660b7ff07e');


--
-- Data for Name: video_tags; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO "public"."video_tags" ("video_id", "tag_id") VALUES
	('550e8400-e29b-41d4-a716-446655441000', '550e8400-e29b-41d4-a716-446655440010'),
	('550e8400-e29b-41d4-a716-446655441000', '550e8400-e29b-41d4-a716-446655440020'),
	('550e8400-e29b-41d4-a716-446655441001', '550e8400-e29b-41d4-a716-446655440011'),
	('550e8400-e29b-41d4-a716-446655441001', '550e8400-e29b-41d4-a716-446655440020'),
	('550e8400-e29b-41d4-a716-446655441002', '550e8400-e29b-41d4-a716-446655440012'),
	('550e8400-e29b-41d4-a716-446655441002', '550e8400-e29b-41d4-a716-446655440021'),
	('550e8400-e29b-41d4-a716-446655441003', '550e8400-e29b-41d4-a716-446655440010'),
	('550e8400-e29b-41d4-a716-446655441003', '550e8400-e29b-41d4-a716-446655440020'),
	('550e8400-e29b-41d4-a716-446655441004', '550e8400-e29b-41d4-a716-446655440011'),
	('550e8400-e29b-41d4-a716-446655441004', '550e8400-e29b-41d4-a716-446655440021'),
	('87fd544b-024c-441a-828d-4cd920ecd138', '4b6b4ca1-5e38-4506-ae70-a1947e1c42ca'),
	('0aa57ba7-b627-45b4-b01d-85a4c4ce0e2b', '550e8400-e29b-41d4-a716-446655440010'),
	('21a62476-e985-4cbd-b976-8f50d78c0b50', '550e8400-e29b-41d4-a716-446655440011'),
	('b4d6190b-4f08-410a-8759-15fd7d732b4b', '550e8400-e29b-41d4-a716-446655440012'),
	('740ca061-4a04-452c-9a8c-0037363125c8', '550e8400-e29b-41d4-a716-446655440010'),
	('b8016822-a058-4a4b-a300-e7efbd31baf8', '550e8400-e29b-41d4-a716-446655440011'),
	('ff3a6103-c1b2-492a-8df4-370b72451d68', '550e8400-e29b-41d4-a716-446655440012'),
	('de3affb5-fd9c-4a4e-9033-a3341f7b695e', '550e8400-e29b-41d4-a716-446655440010'),
	('96e03fff-b705-409a-a686-c22cf8dc19d3', '550e8400-e29b-41d4-a716-446655440011'),
	('6f442888-0fc6-4143-b85e-d5102ab63bfc', '550e8400-e29b-41d4-a716-446655440012'),
	('a6e83156-6fee-48b1-a3a6-809bc51eef8f', '550e8400-e29b-41d4-a716-446655440010'),
	('9878152d-c672-4dae-bbc2-4b8eaffbe184', '550e8400-e29b-41d4-a716-446655440011'),
	('0ce31a5e-75e9-4fd8-bc57-bbe85f396d12', '550e8400-e29b-41d4-a716-446655440012'),
	('6ee24663-93a2-4a7d-8efb-eb099b5e2583', '550e8400-e29b-41d4-a716-446655440010'),
	('95ea2b98-3780-4d80-8090-407d828caa58', '550e8400-e29b-41d4-a716-446655440011'),
	('9f8ddd96-81be-4619-a7e4-6d6a1f852155', '550e8400-e29b-41d4-a716-446655440012'),
	('e51f1cfd-1e90-4aed-bf33-8c292f8d2003', '550e8400-e29b-41d4-a716-446655440010'),
	('8aeb0571-e412-4846-b3e0-f99bb9d524a1', '550e8400-e29b-41d4-a716-446655440011'),
	('afea974d-fe61-4b59-b62c-e0db048fc05c', '550e8400-e29b-41d4-a716-446655440012'),
	('e476db97-414f-454d-a3ed-6816c86fd595', '550e8400-e29b-41d4-a716-446655440010'),
	('d48e46e7-d92f-4a74-85ac-387dad069899', '550e8400-e29b-41d4-a716-446655440011'),
	('44f5c2f6-d4c4-4d98-aee5-1040d9ff9ea5', '550e8400-e29b-41d4-a716-446655440012'),
	('f1221582-31be-43fc-a83a-f85e1631d0e3', '550e8400-e29b-41d4-a716-446655440010'),
	('e5b2f71d-20d2-48ab-a344-63e3553cddcc', '550e8400-e29b-41d4-a716-446655440011'),
	('f0c41f92-5b75-489f-a178-34a565baac39', '550e8400-e29b-41d4-a716-446655440012'),
	('890f36f1-5382-4b3a-98bc-2a121d02859c', '550e8400-e29b-41d4-a716-446655440010'),
	('a76f581e-1dcf-451b-8776-4eb3b22d58a8', '550e8400-e29b-41d4-a716-446655440011'),
	('419a6dc8-c92d-49d5-93a7-5d10f17a0329', '550e8400-e29b-41d4-a716-446655440012'),
	('da2dbbb4-21f6-4a3a-b581-71f971cc454f', '550e8400-e29b-41d4-a716-446655440010'),
	('a140b277-9369-4a9e-a0d7-04a5d2000f91', '550e8400-e29b-41d4-a716-446655440011'),
	('530014c0-8128-4eb2-879b-ef1e2becc4df', '550e8400-e29b-41d4-a716-446655440012'),
	('d8656d7d-f56f-452b-ae07-d1f89facf61b', '550e8400-e29b-41d4-a716-446655440010'),
	('8c42f302-08cc-41fa-ac1b-032661be1a12', '550e8400-e29b-41d4-a716-446655440011'),
	('e269f26e-a036-4b95-afa2-7246b70ddead', '550e8400-e29b-41d4-a716-446655440012'),
	('7c6e67f4-916d-46d7-b2f6-3ded7779e4ec', '550e8400-e29b-41d4-a716-446655440010'),
	('1848141f-2a61-46b3-bda1-e932f9fcbd57', '550e8400-e29b-41d4-a716-446655440011'),
	('1eaaf1a4-5479-4385-8e6f-5facea6e1e79', '550e8400-e29b-41d4-a716-446655440012'),
	('a1dbd6f0-2297-4222-8f05-f3af380334b3', '550e8400-e29b-41d4-a716-446655440010'),
	('b5c236c9-18ec-4fb9-800f-d2befc69b770', '550e8400-e29b-41d4-a716-446655440011'),
	('9d549281-4049-4aa5-9f6f-3e0f948286c4', '550e8400-e29b-41d4-a716-446655440012'),
	('91ed035e-19c7-44d7-9732-b1ada432c9b8', '550e8400-e29b-41d4-a716-446655440010'),
	('1949a2bf-336c-483c-ab71-bc7a6ba688d9', '550e8400-e29b-41d4-a716-446655440011'),
	('d676eaad-e18c-40cd-9700-f657e8b5ba6f', '550e8400-e29b-41d4-a716-446655440012'),
	('bd14a621-fb73-4328-93a5-7c7dce6f6785', '550e8400-e29b-41d4-a716-446655440010'),
	('842b8e45-a7a0-43fd-a97f-bbc08729ba64', '550e8400-e29b-41d4-a716-446655440011'),
	('639e1ea4-7406-4a09-a89c-cf5f0e8e260b', '550e8400-e29b-41d4-a716-446655440012'),
	('26e4101b-c70b-4907-b6ac-9b25aa62d1df', '550e8400-e29b-41d4-a716-446655440010'),
	('5a1865fe-5ec0-4667-bb28-5f21bd38b095', '550e8400-e29b-41d4-a716-446655440011'),
	('86ed453d-766d-4876-a5be-e621786e0e3a', '550e8400-e29b-41d4-a716-446655440012'),
	('a5d7b33c-714b-43ae-ab67-8195d154fabc', '550e8400-e29b-41d4-a716-446655440010'),
	('a9d8280d-9752-491b-a716-086d390528d8', '550e8400-e29b-41d4-a716-446655440011');


--
-- Data for Name: buckets; Type: TABLE DATA; Schema: storage; Owner: supabase_storage_admin
--



--
-- Data for Name: objects; Type: TABLE DATA; Schema: storage; Owner: supabase_storage_admin
--



--
-- Data for Name: s3_multipart_uploads; Type: TABLE DATA; Schema: storage; Owner: supabase_storage_admin
--



--
-- Data for Name: s3_multipart_uploads_parts; Type: TABLE DATA; Schema: storage; Owner: supabase_storage_admin
--



--
-- Data for Name: hooks; Type: TABLE DATA; Schema: supabase_functions; Owner: supabase_functions_admin
--



--
-- Name: refresh_tokens_id_seq; Type: SEQUENCE SET; Schema: auth; Owner: supabase_auth_admin
--

SELECT pg_catalog.setval('"auth"."refresh_tokens_id_seq"', 3, true);


--
-- Name: hooks_id_seq; Type: SEQUENCE SET; Schema: supabase_functions; Owner: supabase_functions_admin
--

SELECT pg_catalog.setval('"supabase_functions"."hooks_id_seq"', 1, false);


--
-- PostgreSQL database dump complete
--

RESET ALL;
